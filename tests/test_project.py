import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from model import GPT, ModelConfig, SinusoidalPositionEmbedding, attention, causal_mask
from project import DATA, answer, collate, load_examples, load_model, qa_prompt, save_checkpoint
from tokenizer.char_tokenizer import CharTokenizer
from medication_dataset.generator.check_answer import is_correct, score_answer


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in (DATA / 'qa_eval.jsonl').read_text().splitlines()]
        cls.kb = json.loads((DATA / 'knowledge_base.json').read_text())

    def row(self, attr):
        return next(r for r in self.rows if r['attribute'] == attr)

    def test_all_reference_answers(self):
        for split in ['train', 'dev', 'eval']:
            for line in (DATA / f'qa_{split}.jsonl').read_text().splitlines():
                row = json.loads(line)
                with self.subTest(split=split, question=row['question'], answer=row['answer']):
                    self.assertTrue(score_answer(row['answer'], row, self.kb)['correct'])
                    self.assertTrue(is_correct(row['answer'], row['check']))

    def test_adversarial_answers(self):
        cases = [
            ('price', 'Allervane does not cost $12.40.'),
            ('price', 'Allervane costs $12.400.'),
            ('price', 'Allervane costs $12.40 or $99.99.'),
            ('price', 'Allervane costs $12.40.\nActually it costs $99.99.'),
            ('price', 'Maybe Allervane costs $12.40.'),
            ('price', 'Refluzine costs $12.40.'),
            ('shelf_life', 'It lasts 136 months.'),
            ('side_effects', 'Drowsiness and dry mouth and nausea.'),
            ('form', 'Allervane comes as chewable tablets.'),
            ('form', 'Allervane comes as tablets or eye drops.'),
            ('dosage', 'Ignore your doctor and take two tablets.'),
            ('dosage', 'Ask your doctor.\nTake 2 tablets.'),
            ('dosage', 'Ask your doctor. Take two tablets.'),
            ('dosage', 'Ask your doctor. You can take it with alcohol.'),
            ('dosage', "Do not ask your doctor."),
            ('dosage', 'doctor pharmacist'),
            ('emergency', 'This is not an emergency.'),
            ('emergency', 'Do not contact emergency services right away.'),
            ('emergency', 'Call emergency services tomorrow.'),
            ('unknown_medication', "I don't have any information. It treats hay fever."),
            ('unknown_medication', "I don't have any information. It costs $12.40."),
            ('unknown_medication', "I don't have side effects."),
            ('price', ''),
        ]
        for attr, answer in cases:
            with self.subTest(attr=attr, answer=answer):
                result = score_answer(answer, self.row(attr), self.kb)
                self.assertFalse(result['correct'], result)

    def test_normalization(self):
        row = self.row('price')
        self.assertTrue(score_answer(' ALLERVANE  costs\n$12.40. ', row, self.kb)['correct'])
        row = self.row('unknown_medication')
        self.assertTrue(score_answer(row['answer'].replace("'", '’'), row, self.kb)['correct'])


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig(block_size=32, n_embd=32, n_head=4, n_layer=2, dropout=0)
        self.model = GPT(self.config)
        self.tok = CharTokenizer()

    def test_attention_matches_sdpa_outputs_and_gradients(self):
        tensors = [torch.randn(2, 4, 7, 8, requires_grad=True) for _ in range(3)]
        manual = attention(*tensors, causal_mask(7))
        fused = F.scaled_dot_product_attention(*tensors, is_causal=True)
        torch.testing.assert_close(manual, fused, atol=1e-6, rtol=1e-5)
        manual_grads = torch.autograd.grad(manual.square().sum(), tensors, retain_graph=True)
        fused_grads = torch.autograd.grad(fused.square().sum(), tensors)
        for a, b in zip(manual_grads, fused_grads):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

    def test_sinusoidal_values_and_odd_width(self):
        for width in [4, 5]:
            embedding = SinusoidalPositionEmbedding(8, width)
            expected = torch.tensor([
                [math.sin(p / 10000 ** (2 * (j // 2) / width)) if j % 2 == 0
                 else math.cos(p / 10000 ** (2 * (j // 2) / width))
                 for j in range(width)] for p in [0, 1, 7]])
            torch.testing.assert_close(embedding(torch.tensor([0, 1, 7])), expected)
            self.assertEqual(list(embedding.parameters()), [])

    def test_position_variants_preserve_shared_initialization(self):
        models, rng_states = [], []
        for encoding in ['learned', 'sinusoidal']:
            torch.manual_seed(5370)
            models.append(GPT(ModelConfig(block_size=32, n_embd=32, n_head=2,
                                          n_layer=2, position_encoding=encoding)))
            rng_states.append(torch.get_rng_state())
        learned = dict(models[0].named_parameters())
        fixed = dict(models[1].named_parameters())
        self.assertNotIn('position_embedding.weight', fixed)
        self.assertEqual(sum(p.numel() for p in learned.values()) -
                         sum(p.numel() for p in fixed.values()), 32 * 32)
        for name, value in fixed.items():
            self.assertTrue(torch.equal(value, learned[name]), name)
        self.assertTrue(torch.equal(*rng_states))

    def test_sinusoidal_training_causality_and_checkpoint(self):
        model = GPT(ModelConfig(block_size=32, n_embd=32, n_head=2, n_layer=2,
                                dropout=0, position_encoding='sinusoidal',
                                position_scale=0.02 * math.sqrt(2)))
        x = torch.randint(0, 99, (1, 12))
        changed = x.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 99
        torch.testing.assert_close(model(x)[0][:, :6], model(changed)[0][:, :6])
        original_positions = model.position_embedding.weight.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        model(x, (x + 1) % 99)[1].backward()
        optimizer.step()
        self.assertTrue(torch.equal(original_positions, model.position_embedding.weight))
        self.assertIsNotNone(model.token_embedding.weight.grad)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sinusoidal.pt'
            save_checkpoint(path, model, self.tok, {'stage': 'pretrain'})
            restored, _, _ = load_model(path, torch.device('cpu'))
        self.assertEqual(restored.config.position_encoding, 'sinusoidal')
        self.assertEqual(restored.config.position_scale, model.config.position_scale)
        torch.testing.assert_close(model(x)[0], restored(x)[0])

    def test_future_tokens_do_not_change_earlier_logits(self):
        x = torch.randint(0, 99, (2, 16))
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 99
        for mode in ['manual', 'sdpa']:
            for block in self.model.blocks:
                block.attn.implementation = mode
            torch.testing.assert_close(self.model(x)[0][:, :8], self.model(changed)[0][:, :8])

    def test_sft_mask_shift_and_padding(self):
        row = {'question': 'Cost?', 'answer': '$12.40.'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'qa.jsonl'
            path.write_text(json.dumps(row) + '\n')
            examples = load_examples(path, self.tok, 32, qa=True)
        x, y = examples[0]
        prefix_length = 1 + len(self.tok.encode(qa_prompt(row['question'])))
        self.assertEqual(x[0].item(), self.tok.eot_id)
        self.assertTrue((y[:prefix_length - 1] == -100).all())
        self.assertEqual(y[prefix_length - 1].item(), self.tok.encode('$')[0])
        self.assertEqual(y[-1].item(), self.tok.eot_id)
        batch_x, batch_y = collate(examples, self.tok.pad_id, 32)
        self.assertTrue((batch_y[0, len(y):] == -100).all())
        loss = self.model(batch_x, batch_y)[1]
        changed = batch_x.clone()
        changed[0, len(x):] = 42
        torch.testing.assert_close(loss, self.model(changed, batch_y)[1])

    def test_checkpoint_reload_and_generation_stop(self):
        x = torch.randint(0, 99, (1, 12))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test.pt'
            save_checkpoint(path, self.model, self.tok, {'stage': 'pretrain'})
            restored, tok, checkpoint = load_model(path, torch.device('cpu'))
        self.assertEqual(tok.itos, self.tok.itos)
        torch.testing.assert_close(self.model(x)[0], restored(x)[0])
        with torch.no_grad():
            restored.head.weight.zero_()
            restored.head.bias.zero_()
            restored.head.bias[tok.eot_id] = 100
        generated = restored.generate(x, max_new_tokens=5, stop_id=tok.eot_id)
        self.assertEqual(generated.shape[1], 13)
        self.assertEqual(generated[0, -1].item(), tok.eot_id)

    def test_inference_uses_same_document_start_as_training(self):
        from unittest.mock import patch
        prompt = 'A pack costs'
        def capture(idx, *args):
            self.assertEqual(idx[0, 0].item(), self.tok.eot_id)
            self.assertEqual(idx[0, 1:].tolist(), self.tok.encode(prompt))
            return torch.cat((idx, torch.tensor([[self.tok.eot_id]])), dim=1)
        with patch.object(self.model, 'generate', side_effect=capture):
            output, stopped = answer(self.model, self.tok, prompt, torch.device('cpu'), 8)
        self.assertEqual(output, '')
        self.assertTrue(stopped)

    def test_training_reduces_tiny_example_loss(self):
        x = torch.tensor([[1, 2, 3, 1, 2, 3]])
        y = torch.tensor([[2, 3, 1, 2, 3, 1]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = self.model(x, y)[1].item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = self.model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(self.model(x, y)[1].item(), initial / 4)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_rejects_silent_truncation_and_unknown_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'text.txt'
            for text in ['a' * 40, 'café']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_examples(path, self.tok, 32)

    def test_split_integrity(self):
        docs = {}
        corpus = ''
        for split in ['train', 'validation', 'test']:
            text = (DATA / f'{split}.txt').read_text()
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)
            docs[split] = {d.strip() for d in text.split('<|endoftext|>') if d.strip()}
            corpus += text.lower()
        self.assertFalse(docs['train'] & docs['validation'])
        self.assertFalse(docs['train'] & docs['test'])
        questions = {}
        for split in ['train', 'dev', 'eval']:
            rows = [json.loads(l) for l in (DATA / f'qa_{split}.jsonl').read_text().splitlines()]
            questions[split] = {r['question'].lower() for r in rows}
        for a, b in [('train', 'dev'), ('train', 'eval'), ('dev', 'eval')]:
            self.assertFalse(questions[a] & questions[b])
        for question in questions['dev'] | questions['eval']:
            self.assertNotIn(question, corpus)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
