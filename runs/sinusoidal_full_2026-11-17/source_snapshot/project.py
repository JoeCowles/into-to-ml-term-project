"""Train, benchmark, evaluate, and run the medication Transformer.

Run `python project.py --help` for the available commands.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch.utils.data import DataLoader

from model import GPT, ModelConfig
from tokenizer.char_tokenizer import CharTokenizer
from medication_dataset.generator.check_answer import VERSION, contains, normalize, score_answer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "medication_dataset" / "combined"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this environment")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this environment")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def precision_context(device):
    # BF16 does not need a GradScaler. Keep MPS/CPU in FP32 for portability.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def encode_checked(tokenizer, text):
    ids = tokenizer.encode(text)
    if tokenizer.unk_id in ids:
        raise ValueError("Input contains non-ASCII characters; normalize it before tokenizing")
    return ids


def qa_prompt(question):
    return f"Q: {question}\nA: "


def load_examples(path, tokenizer, block_size, qa=False):
    """Document-isolated samples. No truncation, cross-split mixing, or prompt loss.

    Right padding cannot affect earlier real tokens under causal attention. Labels
    on padding and (for SFT) the entire prompt are ignored with -100.
    """
    if qa:
        items = read_rows(path)
    else:
        items = [s.strip() for s in Path(path).read_text().split("<|endoftext|>") if s.strip()]
    samples = []
    for item in items:
        if qa:
            prompt = [tokenizer.eot_id] + encode_checked(tokenizer, qa_prompt(item["question"]))
            seq = prompt + encode_checked(tokenizer, item["answer"]) + [tokenizer.eot_id]
            x, y = seq[:-1], seq[1:]
            y[:len(prompt) - 1] = [-100] * (len(prompt) - 1)
        else:
            seq = [tokenizer.eot_id] + encode_checked(tokenizer, item) + [tokenizer.eot_id]
            x, y = seq[:-1], seq[1:]
        if len(x) > block_size:
            raise ValueError(f"{path}: sample length {len(x)} exceeds context {block_size}; increase block_size")
        samples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    if not samples:
        raise ValueError(f"Empty dataset: {path}")
    return samples


def collate(samples, pad_id, block_size):
    # Round to 32 to reduce the number of GPU shapes while keeping padding modest.
    length = min(block_size, math.ceil(max(len(x) for x, _ in samples) / 32) * 32)
    x = torch.full((len(samples), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (xs, ys) in enumerate(samples):
        x[i, :len(xs)], y[i, :len(ys)] = xs, ys
    return x, y


def loader(samples, batch_size, tokenizer, block_size, shuffle=False, seed=5370):
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate, pad_id=tokenizer.pad_id, block_size=block_size))


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    saved_tok = checkpoint["tokenizer"]
    tok = CharTokenizer(saved_tok["chars"], saved_tok["special_tokens"])
    if tok.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint vocabulary and model dimensions disagree")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), tok, checkpoint


def save_checkpoint(path, model, tokenizer, metadata):
    payload = {"format_version": 1, "config": model.config_dict(),
               "tokenizer": {"chars": tokenizer.chars, "special_tokens": tokenizer.special_tokens},
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               **metadata}
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def measure_loss(model, batches, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y in batches:
        count = int((y != -100).sum())
        with precision_context(device):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item() * count
        tokens += count
    return {"loss": total / tokens, "target_tokens": tokens,
            "perplexity": math.exp(min(total / tokens, 80))}


def model_and_tokenizer(config_path):
    config = ModelConfig(**json.loads(Path(config_path).read_text()))
    tok = CharTokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    if config.vocab_size != tok.vocab_size:
        raise ValueError("Set config vocab_size to match the tokenizer")
    return GPT(config), tok


def train(args):
    device = pick_device(args.device)
    seed_all(args.seed)
    sft = args.command == "finetune"
    if sft:
        model, tok, source = load_model(args.checkpoint, device)
        if source["stage"] != "pretrain":
            raise ValueError("Fine-tuning must start from a pretraining checkpoint")
    else:
        model, tok = model_and_tokenizer(args.config)
        model.to(device)
    epochs = args.epochs if args.epochs is not None else (5 if sft else 20)
    batch_size = args.batch_size if args.batch_size is not None else (32 if sft else 8)
    lr = args.lr if args.lr is not None else (1e-4 if sft else 3e-4)
    if epochs < 1 or batch_size < 1 or lr <= 0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs, batch size, learning rate and max steps must be positive")
    train_path = args.data / ("qa_train.jsonl" if sft else "train.txt")
    val_path = args.data / ("qa_dev.jsonl" if sft else "validation.txt")
    samples = load_examples(train_path, tok, model.block_size, sft)
    val_samples = load_examples(val_path, tok, model.block_size, sft)
    batches = loader(samples, batch_size, tok, model.block_size, True, args.seed)
    val_batches = loader(val_samples, batch_size, tok, model.block_size)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ["best.pt", "last.pt", "metrics.jsonl"]):
        raise ValueError(f"{out} already contains a run; choose a new --out directory")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    planned_steps = epochs * len(batches)
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup = min(100, max(1, total_steps // 20))
    meta = {"stage": "finetune" if sft else "pretrain", "seed": args.seed,
            "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": lr,
            "device": str(device), "torch_version": str(torch.__version__),
            "data_hashes": {str(p.name): file_hash(p) for p in (train_path, val_path)},
            "source_checkpoint": str(args.checkpoint) if sft else None,
            "source_sha256": file_hash(args.checkpoint) if sft else None,
            "prompt_format": "<|endoftext|>Q: {question}\nA: ", "answer_only_loss": sft,
            "parameters": sum(p.numel() for p in model.parameters())}
    write_json(out / "config.json", {**meta, "model": model.config_dict()})
    print(f"{meta['stage']} | {device} | {meta['parameters']:,} parameters | {len(samples)} samples | {total_steps} steps", flush=True)
    best, step, started = float("inf"), 0, time.perf_counter()
    initial = measure_loss(model, val_batches, device)
    with (out / "metrics.jsonl").open("w") as log:
        log.write(json.dumps({"epoch": 0, "step": 0, "validation_loss": initial["loss"]}) + "\n")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, count_tokens, batches_done = 0.0, 0, 0
            for x, y in batches:
                fraction = max(0.0, (step - warmup) / max(1, total_steps - warmup))
                multiplier = (step + 1) / warmup if step < warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2
                for group in opt.param_groups:
                    group["lr"] = lr * multiplier
                count = int((y != -100).sum())
                opt.zero_grad(set_to_none=True)
                with precision_context(device):
                    _, loss = model(x.to(device), y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                total_loss += loss.item() * count
                count_tokens += count
                step += 1
                batches_done += 1
                if step % args.log_every == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {total_loss/count_tokens:.4f}", flush=True)
                if step >= total_steps:
                    break
            val = measure_loss(model, val_batches, device)
            record = {"epoch": epoch, "step": step, "train_loss": total_loss / count_tokens,
                      "validation_loss": val["loss"], "validation_perplexity": val["perplexity"],
                      "train_target_tokens": count_tokens, "epoch_complete": batches_done == len(batches),
                      "elapsed_seconds": time.perf_counter() - started}
            log.write(json.dumps(record) + "\n")
            log.flush()
            improved = val["loss"] < best
            best = min(best, val["loss"])
            state = {**meta, **record, "best_validation_loss": best}
            save_checkpoint(out / "last.pt", model, tok, state)
            if improved:
                save_checkpoint(out / "best.pt", model, tok, state)
            print(json.dumps(record), flush=True)
            if step >= total_steps:
                break
    print(f"Saved {out / 'best.pt'} and {out / 'last.pt'}", flush=True)


def answer(model, tok, prompt, device, max_new_tokens, temperature=0.0):
    # Match the document-start token used during BOTH training stages. Omitting
    # this shifts every learned position and creates an avoidable distribution shift.
    ids = [tok.eot_id] + encode_checked(tok, prompt)
    if not prompt or len(ids) > model.block_size:
        raise ValueError("Prompt must be nonempty and fit in the model context")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with precision_context(device):
        generated = model.generate(idx, max_new_tokens, temperature, tok.eot_id, (tok.pad_id, tok.unk_id))
    continuation = generated[0, len(ids):].tolist()
    stopped = bool(continuation and continuation[-1] == tok.eot_id)
    if stopped:
        continuation.pop()
    return tok.decode(continuation), stopped


def wilson(success, total):
    if not total:
        return [0.0, 1.0]
    z, p = 1.96, success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0, center-half), min(1, center+half)]


def evaluate(args):
    device = pick_device(args.device)
    qa_path = args.data / f"qa_{args.split}.jsonl"
    rows = read_rows(qa_path)
    kb = json.loads((args.data / "knowledge_base.json").read_text())
    results = []
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "comparison.json").exists():
        raise ValueError("Evaluation output already exists; choose a new --out directory")
    for checkpoint_path in args.checkpoints:
        seed_all(args.seed)
        model, tok, checkpoint = load_model(checkpoint_path, device)
        model.eval()
        predictions = []
        totals = defaultdict(lambda: [0, 0])
        for i, row in enumerate(rows):
            output, stopped = answer(model, tok, qa_prompt(row["question"]), device, args.max_new_tokens)
            score = score_answer(output, row, kb)
            content_correct = score["correct"]
            # A token-budget cutoff is not a completed answer, even if it contains
            # an expected keyword. Both checkpoints receive the same limit.
            if not stopped:
                score["reasons"].append("generation_limit_reached")
                score["correct"] = False
            totals[row["attribute"]][0] += int(score["correct"])
            totals[row["attribute"]][1] += 1
            predictions.append({**row, "prediction": output, "stopped_at_eot": stopped,
                                "content_rule_correct": content_correct, **score})
            if (i + 1) % 25 == 0:
                print(f"{checkpoint_path}: {i+1}/{len(rows)} questions", flush=True)
        correct = sum(v[0] for v in totals.values())
        metrics = {attr: {"correct": good, "total": n, "accuracy": good/n,
                           "wilson_95": wilson(good, n)} for attr, (good, n) in sorted(totals.items())}
        summary = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
                   "stage": checkpoint["stage"], "scorer_version": VERSION, "split": args.split,
                   "questions_sha256": file_hash(qa_path), "knowledge_base_sha256": file_hash(args.data / "knowledge_base.json"),
                   "prompt_format": "<|endoftext|>Q: {question}\nA: ", "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
                   "total": len(rows), "correct": correct, "accuracy": correct/len(rows),
                   "completed_answers": sum(r["stopped_at_eot"] for r in predictions),
                   "content_rule_accuracy": sum(r["content_rule_correct"] for r in predictions)/len(rows),
                   "macro_attribute_accuracy": sum(v["accuracy"] for v in metrics.values())/len(metrics),
                   "wilson_95": wilson(correct, len(rows)), "by_attribute": metrics,
                   "uncertainty_note": "Wilson intervals treat questions as independent; shared templates/medications violate that assumption. Descriptive only.",
                   "scoring_note": "Accuracy requires full-answer rule checks AND end-of-text termination. content_rule_accuracy omits only the termination requirement. Neither is semantic ground truth; manually audit predictions."}
        # Evaluate the pretraining test corpus only when explicitly requested.
        if args.corpus_test:
            examples = load_examples(args.data / "test.txt", tok, model.block_size)
            summary["corpus_test"] = measure_loss(model, loader(examples, args.batch_size, tok, model.block_size), device)
        name = f"{len(results)+1}_{checkpoint_path.parent.name}_{checkpoint_path.stem}"
        if args.probes:
            probe_predictions, probe_totals = [], defaultdict(lambda: [0, 0])
            for probe in read_rows(args.data / "probes.jsonl"):
                output, stopped = answer(model, tok, probe["prefix"], device, args.max_new_tokens)
                # A completion probe asks whether the value occurs in the first
                # completed sentence. This is a weaker metric than QA correctness.
                first_sentence = re.split(r"[.!?](?:\s|$)", output, maxsplit=1)[0]
                good = contains(normalize(first_sentence), probe["expected"])
                probe_totals[probe["kind"]][0] += int(good)
                probe_totals[probe["kind"]][1] += 1
                probe_predictions.append({**probe, "prediction": output, "expected_value_found": good})
            summary["probes"] = {kind: {"correct": good, "total": n, "value_match_rate": good/n}
                                 for kind, (good, n) in probe_totals.items()}
            summary["probe_note"] = "Expected-value match in the first continuation sentence; does not detect all contradictions. Inspect predictions."
            summary["probes_sha256"] = file_hash(args.data / "probes.jsonl")
            (args.out / f"{name}_probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probe_predictions))
        (args.out / f"{name}_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
        write_json(args.out / f"{name}_metrics.json", summary)
        results.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        del model
    write_json(args.out / "comparison.json", results)


def benchmark(args):
    device = pick_device(args.device)
    seed_all(args.seed)
    model, tok = model_and_tokenizer(args.config)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    result = {"device": str(device), "torch_version": str(torch.__version__),
              "precision": "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32",
              "parameters": sum(p.numel() for p in model.parameters()), "config": model.config_dict(),
              "warmup_steps": args.warmup, "measured_steps": args.steps}
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    for stage, filename, batch_size, epochs, val_name in [
        ("pretrain", "train.txt", args.batch_size, args.pretrain_epochs, "validation.txt"),
        ("finetune", "qa_train.jsonl", args.qa_batch_size, args.finetune_epochs, "qa_dev.jsonl")]:
        samples = load_examples(args.data / filename, tok, model.block_size, stage == "finetune")
        batches = loader(samples, batch_size, tok, model.block_size, True, args.seed)
        iterator = iter(batches)
        def next_batch():
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(batches)
                return next(iterator)
        model.train()
        padded_tokens, target_tokens = 0, 0
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                sync(device)
                start = time.perf_counter()
            x, y = next_batch()
            count = int((y != -100).sum())
            opt.zero_grad(set_to_none=True)
            with precision_context(device):
                _, loss = model(x.to(device), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            loss.item()  # Same host synchronization as the training loop.
            if i >= args.warmup:
                padded_tokens += x.numel()
                target_tokens += count
        sync(device)
        elapsed = time.perf_counter() - start
        validation = load_examples(args.data / val_name, tok, model.block_size, stage == "finetune")
        sync(device)
        val_start = time.perf_counter()
        measure_loss(model, loader(validation, batch_size, tok, model.block_size), device)
        sync(device)
        val_seconds = time.perf_counter() - val_start
        step_seconds = elapsed / args.steps
        result[stage] = {"batch_size": batch_size, "samples": len(samples), "steps_per_epoch": len(batches),
                         "seconds_per_step": step_seconds, "padded_tokens_per_second": padded_tokens / elapsed,
                         "target_tokens_per_second": target_tokens / elapsed, "validation_seconds": val_seconds,
                         "epochs": epochs, "estimated_seconds": step_seconds * len(batches) * epochs + val_seconds * (epochs + 1)}
        print(json.dumps({stage: result[stage]}, indent=2), flush=True)
    result["note"] = "Short local benchmark, random initial weights. Includes optimizer, clipping, data transfer and validation; excludes checkpoint writes, startup and autoregressive QA evaluation. Not a convergence prediction."
    write_json(args.out, result)
    print(f"Saved benchmark to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("pretrain", "finetune", "benchmark", "evaluate", "generate"):
        p = sub.add_parser(command)
        p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        p.add_argument("--seed", type=int, default=5370)
        if command != "generate":
            p.add_argument("--data", type=Path, default=DATA)
        if command in {"pretrain", "benchmark"}:
            p.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
        if command in {"finetune", "generate"}:
            p.add_argument("--checkpoint", type=Path, required=True)
        if command in {"pretrain", "finetune"}:
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--epochs", type=int)
            p.add_argument("--batch-size", type=int)
            p.add_argument("--lr", type=float)
            p.add_argument("--max-steps", type=int, help="Cap optimizer steps for a smoke test or short experiment")
            p.add_argument("--log-every", type=int, default=50)
        elif command == "benchmark":
            p.add_argument("--out", type=Path, default=ROOT / "runs" / "benchmark.json")
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--qa-batch-size", type=int, default=32)
            p.add_argument("--warmup", type=int, default=5)
            p.add_argument("--steps", type=int, default=30)
            p.add_argument("--pretrain-epochs", type=int, default=20)
            p.add_argument("--finetune-epochs", type=int, default=5)
        elif command == "evaluate":
            p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--split", choices=["dev", "eval"], default="eval")
            p.add_argument("--max-new-tokens", type=int, default=192)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--corpus-test", action="store_true")
            p.add_argument("--probes", action="store_true", help="Also evaluate familiar/unseen completion probes")
        elif command == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--qa", action="store_true", help="Wrap prompt in the same Q/A format as SFT")
            p.add_argument("--max-new-tokens", type=int, default=192)
            p.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.command in {"pretrain", "finetune"}:
        if args.log_every < 1:
            parser.error("--log-every must be positive")
        train(args)
    elif args.command == "benchmark":
        if min(args.steps, args.batch_size, args.qa_batch_size) < 1 or args.warmup < 0:
            parser.error("Invalid benchmark sizes")
        benchmark(args)
    elif args.command == "evaluate":
        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be positive")
        evaluate(args)
    else:
        seed_all(args.seed)
        device = pick_device(args.device)
        model, tok, _ = load_model(args.checkpoint, device)
        prompt = qa_prompt(args.prompt) if args.qa else args.prompt
        output, stopped = answer(model, tok, prompt, device, args.max_new_tokens, args.temperature)
        print(output)
        if not stopped:
            print("[generation reached token limit]")


if __name__ == "__main__":
    main()
