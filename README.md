# Medication Support Desk Transformer

A character-level decoder-only Transformer for the CS 5370 term project.
The domain and all medication facts are fictional.

The latest full run uses 3,207,267 parameters, two attention heads, and scaled
sinusoidal positions. Fine-tuned rule-based QA accuracy is 115/248 (46.4%), compared
with 88/248 (35.5%) for the original four-head learned-position model. These are
reused evaluation questions, and manual review found false positives in the scorer.
See [RESULTS.md](RESULTS.md) for checkpoints, comparisons, and limitations.

The Transformer uses combined learned Q/K/V
projections, causal multi-head attention, learned token embeddings, configurable
learned or fixed sinusoidal positions,
pre-normalized residual blocks, a GELU feed-forward network, and an LM head.
Run `project.py` to train, evaluate, or generate text with this model.

## Setup

Python 3.12 is the tested version. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Alternatively, `uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python -r requirements.txt`.
The local `.venv` has already been installed. `requirements-macos-lock.txt`
records the exact packages used for the Mac verification. CUDA installations
should use the appropriate PyTorch build for their driver rather than the Mac lock.

The program automatically selects CUDA, then Apple MPS, then CPU. Use
`--device mps` to require the Mac GPU. Training on CPU is substantially slower.

## Train and fine-tune

```bash
python project.py pretrain --device mps --out runs/pretrain
python project.py finetune --device mps \
  --checkpoint runs/pretrain/best.pt --out runs/finetune
```

These commands reproduce the original architecture. For the full sinusoidal
configuration, use fresh output directories:

```bash
python project.py pretrain --device mps --config configs/sinusoidal.json \
  --epochs 20 --out runs/sinusoidal_pretrain_new
python project.py finetune --device mps --epochs 5 \
  --checkpoint runs/sinusoidal_pretrain_new/best.pt --out runs/sinusoidal_finetune_new
python project.py evaluate --device mps \
  --checkpoints runs/sinusoidal_pretrain_new/best.pt runs/sinusoidal_finetune_new/best.pt \
  --out runs/sinusoidal_evaluation_new --corpus-test --probes
```

The completed run is in `runs/sinusoidal_full_2026-11-17/`, including both selected
checkpoints, predictions, completion probes, figures, source snapshots, and a run
manifest. Its config is `configs/sinusoidal.json`; the original defaults remain in
`configs/model.json` for reproducing the baseline.

Original baseline defaults:

| Setting | Value |
|---|---|
| Parameters | 3,469,411 |
| Decoder blocks | 4 |
| Attention heads | 4 |
| Embedding width | 256 |
| Maximum context | 1,024 character tokens |
| Vocabulary | 99 tokens |
| Dropout | 0.1 |
| Pretraining | 20 epochs, batch 8, peak learning rate 0.0003 |
| Fine-tuning | 5 epochs, batch 32, peak learning rate 0.0001 |
| Optimizer | AdamW; weight decay 0.01; gradient norm clipped to 1.0 |
| LR schedule | Linear warmup followed by cosine decay to 10% of peak |
| Mac precision | Float32 |
| CUDA precision | BF16 autocast when supported; otherwise float32 |

Change architecture in `configs/model.json`; pass a different file with `--config`.
Position encoding defaults to `"position_encoding": "learned"`. Set it to
`"sinusoidal"` for fixed sine/cosine positions (262,144 fewer trainable parameters
at the default width/context). `"position_scale"` defaults to 1.0 and multiplies
the position vectors before addition to token embeddings. For a comparison that
approximately matches the learned positions' initial RMS of 0.02, use sinusoidal
scale `0.028284271247461905`; unit-scale sinusoids have RMS about 0.707. Token
embeddings are not additionally scaled in either variant. Shared weights and RNG
state initialize identically across position variants for the same seed. The
context limit remains unchanged, and checkpoints record both settings. Existing
checkpoints without these fields load with the original learned-position behavior.
Change training length/batch size/rate using `--epochs`, `--batch-size`, `--lr`.
`--max-steps 2` is useful for a smoke test, not a finished model.
Use a new output directory for every run: existing runs are protected from overwrite.
Training does not currently resume interrupted optimizer state; `last.pt` remains
usable for inference, and completed epoch checkpoints are saved throughout the run.

Each epoch shuffles and visits every training document or QA row once. Documents
stay isolated from each other, with a start/end `<|endoftext|>` token. Samples are
right-padded to the longest sample in the batch (rounded to 32). Padding labels
are ignored; causal attention prevents padding from influencing earlier tokens.
Oversize samples raise an error rather than silently truncating facts. All current
documents fit in 1,024 tokens.

Pretraining uses only `combined/train.txt`, selecting checkpoints by full
`validation.txt` token-weighted loss. Fine-tuning uses only `qa_train.jsonl`,
selecting by full `qa_dev.jsonl` answer-token loss. Its serialized example is:

```text
<|endoftext|>Q: What does Allervane cost?
A: A pack of Allervane costs $12.40.<|endoftext|>
```

Fine-tuning masks question/prompt labels and padding; the model learns answer
characters and the end token. `qa_eval.jsonl` and `test.txt` never enter training
or checkpoint selection. Dev has no unknown-medication or emergency rows, so
its loss cannot select those behaviors specifically.

Inference automatically prepends the same document-start token used in both
training stages. Do not type that token yourself in `--prompt`.

Each output directory contains:

- `best.pt`: lowest validation-loss trained checkpoint for that stage.
- `last.pt`: final completed save, useful if a run is interrupted later.
- `config.json`: architecture, training settings, seed, dataset hashes, provenance.
- `metrics.jsonl`: initial validation loss and per-epoch train/validation losses,
  perplexity, elapsed time, and completed/partial epoch flags.

Checkpoints bundle the weights, model configuration, and full tokenizer mapping.
Inference does not need the training corpus or a separate tokenizer file.
Retain `project.py`, `model.py`, the `tokenizer` package, and the evaluation module
when distributing the runnable project. Seeded runs are repeatable within usual
backend limits; bitwise equivalence across CPU/MPS/CUDA is not promised.

## Generate answers or text

```bash
python project.py generate --checkpoint runs/finetune/best.pt --qa \
  --prompt "What does Allervane cost?"

python project.py generate --checkpoint runs/pretrain/best.pt \
  --prompt "Allervane is used to treat" --temperature 0.7
```

Temperature zero is greedy decoding. Generation stops at the end token or the
requested token limit; reaching the limit is reported. Prompt characters must be
in the tokenizer's ASCII vocabulary: normalize curly quotes and accented letters
before use. The model never retrieves answers from the knowledge base at inference.

## Evaluate fairly

Use dev questions while developing. Run the final comparison on the same reserved
questions for both checkpoints, using the same prompt format and greedy decoding:

```bash
python project.py evaluate --device mps \
  --checkpoints runs/pretrain/best.pt runs/finetune/best.pt \
  --out runs/evaluation --corpus-test --probes
```

`--split dev` uses development questions instead. `--corpus-test` adds held-out
corpus loss; `--probes` adds familiar/unseen sentence completion tests. Once the
reserved evaluation has been inspected, do not tune to its failures and then
present the rerun as an untouched test. Use fresh held-out questions for further
model development.

Evaluation saves every generated answer, reference, question, and failure reason,
plus overall accuracy, accuracy by attribute, and macro-average attribute accuracy.
Token-limit cutoffs count as failures. Full answers are scored; text after the first
line is not silently discarded. Dataset/checkpoint hashes identify the exact comparison.
Completed-answer counts and content-rule accuracy without the termination requirement
are also reported, so improvements in response formatting can be distinguished
from improvements in answering facts.
Wilson intervals are descriptive only: questions share templates and medications,
so they do not represent independent real-world samples.

Export the loss curves and evaluation chart after training:

```bash
python plot_results.py --pretrain runs/pretrain/metrics.jsonl \
  --finetune runs/finetune/metrics.jsonl \
  --comparison runs/evaluation/comparison.json --out runs/figures
```

The strengthened scorer checks phrase boundaries, numeric conflicts, negations,
other medication names, conflicting known values, explicit referrals and emergency
help-seeking, and obvious dosing instructions written as words. It is deliberately
conservative, not a semantic judge or evidence of clinical safety. Correct novel
paraphrases can fail and unanticipated contradictions can pass. Manually inspect
saved correct and incorrect predictions and report these limitations. Completion
probes use a separately labeled, weaker expected-value match in the first generated
sentence; they are not counted as QA accuracy.

Training already includes FAQ/chat documents. The before/after comparison therefore
measures the benefit of focused QA fine-tuning, not first exposure to questions.
The 2,804 fine-tuning pairs contain 970 distinct questions. Only two reserved
emergency examples exist, so their score supports very limited conclusions.

## Timing on this Mac

Measured on an Apple M4 Pro, 20 GPU cores, 48 GB memory, with PyTorch 2.15.0/MPS.
`benchmarks/m4_pro.json` records the measurement and exact configuration.

- Short benchmark: 0.0844 seconds per pretraining step and 0.0438 per fine-tuning step.
- Full-epoch check: 450 pretraining steps plus initial/final validation took about 34 seconds.
- Projection: roughly 11–13 minutes for 20 pretraining epochs; about 20 seconds for
  five fine-tuning epochs, before startup and checkpoint writes.
- Practical allowance: 15–25 minutes including checkpoint writes and generation-based
  evaluation. Thermal load, other applications and answer lengths can change this.

These are runtime measurements, not a guarantee that 20 epochs is sufficient to
learn the task. Choose duration using validation behavior and document failures.
Reproduce timing with:

```bash
python project.py benchmark --device mps --out runs/benchmark_new.json
```

## Implementation and verification

`model.py` includes explicit `attention()` and `causal_mask()` implementations.
The default uses PyTorch scaled-dot-product attention for
speed while keeping all model components/projections here. Set `attention` to
`manual` in the configuration for the explicit matrix-operation path. Tests compare
both outputs and gradients and verify that future tokens cannot affect earlier logits.

Tests also cover all 3,124 reference QA answers, adversarial scorer examples,
full-response checks, split separation, answer-only loss masks, padding invariance,
checkpoint reload, generation termination, matching document-start tokens between
training and inference, and learning a small controlled sequence.
The smoke runs under `runs/smoke_*` are software checks, not final project results.

The tokenizer statistics script now runs from the repository root:

```bash
python -m tokenizer.train_tokenizer
```

Dataset generation remains documented in `medication_dataset/README.md`. The data
was not regenerated during the implementation. The root `runs/` directory and
virtual environment are ignored by Git; explicitly package the two final checkpoints
when preparing the class submission.

AI assistance was used to develop the model code, implement the training and
evaluation pipeline, add checks, and draft these instructions. Include this work
alongside the team's dataset-generation assistance in the report's AI Use Statement.
