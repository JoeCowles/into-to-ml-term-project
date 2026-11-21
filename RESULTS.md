# Training results

## Full sinusoidal rerun — November 17, 2026

The full pipeline was rerun from scratch with **two attention heads and fixed sinusoidal positions scaled by 0.028284271247461905**. Width remains 256, context 1,024, and depth four blocks. The model has **3,207,267 trainable parameters**, 262,144 (7.6%) fewer than the original model. Data and tokenizer were preserved, and dataset hashes were checked against the original runs.

Fine-tuned strict rule-based QA accuracy increased from **88/248 (35.5%) to 115/248 (46.4%)**, a gain of 27 answers or 10.9 percentage points. This is an improvement under the unchanged scorer, not a fully audited semantic accuracy estimate: manual review identified accepted answers with incorrect medication names. Generalization remains weak, particularly for side-effect questions.

This full comparison changes **both** head count (four to two) and positional encoding. It does not isolate the effect of sinusoidal positions alone. The preceding three-epoch position experiment held heads at two and is saved in [the position comparison](runs/position_comparison_2026-11-17/comparison.json).

### Training and checkpoint selection

| Stage | Epochs run | Selected epoch | Training + validation runtime | Selection loss |
|---|---:|---:|---:|---:|
| Pretraining | 20 | 17 | 512.0 seconds | 0.06024 |
| Fine-tuning | 5 | 2 | 15.3 seconds | 0.04850 |

The batch sizes (8/32), peak learning rates (0.0003/0.0001), seed (5370), AdamW settings, clipping, full-stage warmup/cosine schedules, prompt format, and selection datasets match the original run. Pretraining uses document validation loss; fine-tuning uses QA-dev answer-token loss. The selected checkpoints are not necessarily the final epochs.

Pretraining was 28.1% faster than the original 712.0-second run. Its best document-validation loss was 3.4% lower (0.06024 versus 0.06236), while fine-tuning's best QA-dev loss was slightly worse (0.04850 versus 0.04473). The large early-learning advantage observed in short experiments narrowed during full training. Whole-pipeline wall time, including QA evaluation, completion probes, and figure generation, was 714.1 seconds (about 12 minutes); manual review and report preparation are additional.

- [Pretrained checkpoint](runs/sinusoidal_full_2026-11-17/pretrain/best.pt)
- [Fine-tuned checkpoint](runs/sinusoidal_full_2026-11-17/finetune/best.pt)
- [Architecture configuration](configs/sinusoidal.json)
- [Manifest: commands, source/data hashes, timings, and selected checkpoints](runs/sinusoidal_full_2026-11-17/run_manifest.json)

### Reused evaluation comparison

Both architectures were scored using the same 248 reserved questions, greedy decoding, 192-character-token limit, prompt format, and version 2.0 scorer. The questions had already been inspected in earlier work, so this is **reused evaluation**, not a fresh untouched test. New checkpoints were selected using validation/dev losses only; no evaluation questions entered training.

| Metric | Original pretrained | Sinusoidal pretrained | Original fine-tuned | Sinusoidal fine-tuned |
|---|---:|---:|---:|---:|
| Strict complete-answer accuracy | 2/248 (0.8%) | 1/248 (0.4%) | 88/248 (35.5%) | 115/248 (46.4%) |
| Content-rule accuracy without termination requirement | 8.1% | 0.4% | 35.5% | 46.4% |
| Answers terminating within the limit | 18/248 | 1/248 | 248/248 | 248/248 |
| Held-out document loss | 1.6553 | 1.6350 | 1.8322 | 1.6382 |

Fine-tuned performance by attribute:

| Attribute | Original correct | Sinusoidal correct | Sinusoidal accuracy |
|---|---:|---:|---:|
| dosage | 12/36 | 18/36 | 50.0% |
| emergency | 1/2 | 1/2 | 50.0% |
| form | 22/36 | 21/36 | 58.3% |
| price | 12/36 | 20/36 | 55.6% |
| shelf life | 24/36 | 24/36 | 66.7% |
| side effects | 0/36 | 1/36 | 2.8% |
| treats | 12/36 | 19/36 | 52.8% |
| unknown medication | 5/30 | 11/30 | 36.7% |

The pretrained checkpoint still performs poorly on bounded QA despite learning document completions. Fine-tuning improves answer termination and task behavior. Side effects remain a major failure: only 1/36 succeeds under the scorer. Emergency results contain only two questions.

### Completion probes

All 312 probes were run for both new checkpoints. These measure whether the expected value appears in the first generated sentence, a weaker criterion than strict QA scoring. They do not check all contradictions or require answer termination. The original full run did not run this suite, so no original-model probe improvement is claimed.

| Checkpoint | Familiar wording | Unseen wording |
|---|---:|---:|
| pretrain | 101/144 (70.1%) | 84/168 (50.0%) |
| finetune | 112/144 (77.8%) | 89/168 (53.0%) |

### Manual review and scorer limitations

The scorer was kept unchanged to preserve comparison with the original run. A targeted review found **11 scorer-accepted answers in the sinusoidal fine-tuned run containing another listed medication name**, compared with four in the original fine-tuned run. The name check knows active knowledge-base medications but misses some invented unknown-name substitutions. For example, a question about Lunovex was accepted despite an answer saying it had no information about Umbrivane.

Five accepted dosage answers in the new run also incorrectly deny having information about a known medication while giving a valid referral. This satisfies the current dosage rules but does not make the entire response factually sound. These flags are documented rather than silently changing the scorer after evaluation. They are not a comprehensive manual regrading, and **115/248 must be reported as rule-based accuracy**.

The saved generation examples illustrate the remaining language gap:

- `What does Allervane cost?` → `Allervane costs $12.40 per pack.`
- `What would I spend on Allervane?` → `I can't give dosing advice for Allervane. Please ask your doctor or pharmacist.`

Both answers terminate, but the second answers the wrong kind of question. One training seed, template-driven data, familiar-wording validation, reused evaluation, and the rule-based metric limit the conclusions.

### Verification and artifacts

The sinusoidal implementation passed 14 regression tests, including fixed-position values, shared initialization, causality, training, and checkpoint reload. Full-run checkpoint hashes, configuration, epoch/step counts, data hashes, fine-tuning source provenance, prediction counts, and probe counts were verified. Every saved QA prediction was rescored with the unchanged scorer and reproduced its recorded outcome.

- [Full before/after evaluation](runs/sinusoidal_full_2026-11-17/evaluation/comparison.json)
- [Original-versus-rerun metrics](runs/sinusoidal_full_2026-11-17/baseline_comparison.csv)
- [Fine-tuned predictions and failure reasons](runs/sinusoidal_full_2026-11-17/evaluation/2_finetune_best_predictions.jsonl)
- [Targeted manual review with examples](runs/sinusoidal_full_2026-11-17/manual_review.json)
- [Generation examples](runs/sinusoidal_full_2026-11-17/generation_examples.json)
- [Loss curves](runs/sinusoidal_full_2026-11-17/figures/loss_curves.png)
- [New pretraining versus fine-tuning QA chart](runs/sinusoidal_full_2026-11-17/figures/qa_comparison.png)
- [Original-versus-sinusoidal QA chart](runs/sinusoidal_full_2026-11-17/figures/baseline_comparison.png)
- [Verification record](runs/sinusoidal_full_2026-11-17/verification.json)

The original run and its checkpoints are preserved below and in their original directories. Use the sinusoidal configuration explicitly to reproduce the new experiment; the baseline configuration remains unchanged.

---

## Original learned-position baseline — November 16, 2026

The 3,469,411-parameter Transformer was trained on the M4 Pro Mac using MPS. Both checkpoints are runnable. The final model is an experimental baseline: it learns familiar factual patterns and short-answer formatting, but generalization to unseen question phrasings remains weak.

### Training and selected checkpoints

| Stage | Epochs run | Selected epoch | Runtime | Selection loss |
|---|---:|---:|---:|---:|
| pretrain | 20 | 16 | 712.0 seconds | 0.06236 |
| finetune | 5 | 2 | 19.1 seconds | 0.04473 |

Pretraining selection uses document validation loss. Fine-tuning selection uses answer-token loss on QA dev; these two losses are different objectives and should not be compared as the same metric. Runtime includes training and validation and excludes final checkpoint writes and generation-based evaluation.

- Pretrained checkpoint: [runs/pretrain/best.pt](runs/pretrain/best.pt).
- Fine-tuned checkpoint: [runs/finetune/best.pt](runs/finetune/best.pt).
- Exact file hashes and selected epochs: [run manifest](runs/run_manifest.json).

### Final held-out comparison

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

### Observed examples

Final fine-tuned model, greedy decoding:

- `What does Allervane cost?` → `A pack of Allervane costs $12.40.`
- `What would I spend on Allervane?` → `Allervane is used to treat seasonal allergies and hay fever.`

The second answer supplies a different attribute. The words `spend`, `budget`, `unwell`, `adult`, `bigger`, and `doses` occur zero times as whole words in both the pretraining corpus and QA training text, despite appearing in evaluation questions. This is evidence of a vocabulary/phrasing coverage gap. A randomly initialized model has no general-language pretraining to supply those meanings. That gap is a plausible contributor to the failures; this run does not isolate it experimentally.

For a future experiment, broaden training language while keeping the facts fixed and reserve new evaluation formulations before further development. Do not train on these evaluation questions or select models using their scores. More epochs alone have not been shown to solve the language-coverage problem.

### Verification and limitations

- Eleven regression tests pass, covering causal attention, matching manual/fused outputs and gradients, padding and QA loss masks, checkpoint loading, the shared start-token convention, split separation, and a small learning test.
- All 3,124 reference QA answers pass the revised scorer. Adversarial examples include negated facts, conflicting prices, different medications, extra-line instructions and word-form dosage amounts.
- The scorer remains conservative and rule-based. Valid paraphrases can fail; unanticipated contradictions can pass. Saved predictions support manual review.
- The corpus is synthetic and template-driven. Validation uses training wording; the test corpus uses held-out wording and a different document-type mix. Low validation loss is not evidence of broad language competence.
- A preliminary evaluation exposed a missing document-start token in inference. The shared training/inference format was corrected, fine-tuning was repeated with unchanged hyperparameters, and the same evaluation was rerun. Earlier runs are archived under `runs/finetune_pre_bos_fix` and `runs/evaluation_pre_bos_fix`; only `runs/evaluation` contains the final comparison. No reserved questions were added to training.
- The full optional completion-probe suite was not run for these results; the command supports it. The archived generation examples are illustrations, not a probe accuracy estimate.

### Artifacts

- [Final metrics](runs/evaluation/comparison.json)
- [Pretrained predictions and failure reasons](runs/evaluation/1_pretrain_best_predictions.jsonl)
- [Fine-tuned predictions and failure reasons](runs/evaluation/2_finetune_best_predictions.jsonl)
- [Generation examples](runs/generation_examples.json)
- [Loss curves](runs/figures/loss_curves.png)
- [QA comparison chart](runs/figures/qa_comparison.png)
- [Running instructions](README.md)

Everything in the medication domain is fictional. These results are for the class experiment.
