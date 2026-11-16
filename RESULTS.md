# Local training results — November 16, 2026

The 3,469,411-parameter Transformer was trained on the M4 Pro Mac using MPS. Both checkpoints are runnable. The final model is an experimental baseline: it learns familiar factual patterns and short-answer formatting, but generalization to unseen question phrasings remains weak.

## Training and selected checkpoints

| Stage | Epochs run | Selected epoch | Runtime | Selection loss |
|---|---:|---:|---:|---:|
| pretrain | 20 | 16 | 712.0 seconds | 0.06236 |
| finetune | 5 | 2 | 19.1 seconds | 0.04473 |

Pretraining selection uses document validation loss. Fine-tuning selection uses answer-token loss on QA dev; these two losses are different objectives and should not be compared as the same metric. Runtime includes training and validation and excludes final checkpoint writes and generation-based evaluation.

- Pretrained checkpoint: [runs/pretrain/best.pt](runs/pretrain/best.pt).
- Fine-tuned checkpoint: [runs/finetune/best.pt](runs/finetune/best.pt).
- Exact file hashes and selected epochs: [run manifest](runs/run_manifest.json).

## Final held-out comparison

Both models received the same 248 reserved questions, the same document-start token and `Q: ... / A: ...` prompt, greedy decoding, and a limit of 192 generated character tokens. Correctness requires the full answer to pass the stricter rule-based checks and terminate with the end token.

| Metric | Pretrained | Fine-tuned |
|---|---:|---:|
| Strict complete-answer accuracy | 2/248 (0.8%) | 88/248 (35.5%) |
| Full-answer content checks, without termination requirement | 8.1% | 35.5% |
| Answers that terminated within the limit | 18/248 | 248/248 |
| Held-out document loss (nats/token) | 1.6553 | 1.8322 |

| Attribute | Fine-tuned correct | Accuracy |
|---|---:|---:|
| dosage | 12/36 | 33.3% |
| emergency | 1/2 | 50.0% |
| form | 22/36 | 61.1% |
| price | 12/36 | 33.3% |
| shelf life | 24/36 | 66.7% |
| side effects | 0/36 | 0.0% |
| treats | 12/36 | 33.3% |
| unknown medication | 5/30 | 16.7% |

Fine-tuning clearly improves response formatting. The model still confuses the requested attribute and sometimes substitutes a different medication. Zero strict side-effect successes and only five successful unknown-medication responses are substantive failures, not a reliable assistant. The emergency category contains only two questions, so its percentage is especially weak evidence.

The pretrained model can reproduce facts when prompted with a familiar product-label prefix. Its poor QA score does not mean it learned no facts; the QA task also requires interpreting an unfamiliar question and producing a bounded response.

## Observed examples

Final fine-tuned model, greedy decoding:

- `What does Allervane cost?` → `A pack of Allervane costs $12.40.`
- `What would I spend on Allervane?` → `Allervane is used to treat seasonal allergies and hay fever.`

The second answer supplies a different attribute. The words `spend`, `budget`, `unwell`, `adult`, `bigger`, and `doses` occur zero times as whole words in both the pretraining corpus and QA training text, despite appearing in evaluation questions. This is evidence of a vocabulary/phrasing coverage gap. A randomly initialized model has no general-language pretraining to supply those meanings. That gap is a plausible contributor to the failures; this run does not isolate it experimentally.

For a future experiment, broaden training language while keeping the facts fixed and reserve new evaluation formulations before further development. Do not train on these evaluation questions or select models using their scores. More epochs alone have not been shown to solve the language-coverage problem.

## Verification and limitations

- Eleven regression tests pass, covering causal attention, matching manual/fused outputs and gradients, padding and QA loss masks, checkpoint loading, the shared start-token convention, split separation, and a small learning test.
- All 3,124 reference QA answers pass the revised scorer. Adversarial examples include negated facts, conflicting prices, different medications, extra-line instructions and word-form dosage amounts.
- The scorer remains conservative and rule-based. Valid paraphrases can fail; unanticipated contradictions can pass. Saved predictions support manual review.
- The corpus is synthetic and template-driven. Validation uses training wording; the test corpus uses held-out wording and a different document-type mix. Low validation loss is not evidence of broad language competence.
- A preliminary evaluation exposed a missing document-start token in inference. The shared training/inference format was corrected, fine-tuning was repeated with unchanged hyperparameters, and the same evaluation was rerun. Earlier runs are archived under `runs/finetune_pre_bos_fix` and `runs/evaluation_pre_bos_fix`; only `runs/evaluation` contains the final comparison. No reserved questions were added to training.
- The full optional completion-probe suite was not run for these results; the command supports it. The archived generation examples are illustrations, not a probe accuracy estimate.

## Artifacts

- [Final metrics](runs/evaluation/comparison.json)
- [Pretrained predictions and failure reasons](runs/evaluation/1_pretrain_best_predictions.jsonl)
- [Fine-tuned predictions and failure reasons](runs/evaluation/2_finetune_best_predictions.jsonl)
- [Generation examples](runs/generation_examples.json)
- [Loss curves](runs/figures/loss_curves.png)
- [QA comparison chart](runs/figures/qa_comparison.png)
- [Running instructions](README.md)

Everything in the medication domain is fictional. These results are for the class experiment.
