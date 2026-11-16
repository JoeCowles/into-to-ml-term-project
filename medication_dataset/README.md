# Medication Support Desk dataset (CS 5370 term project)

**Fictional domain:** *Halverin Medical Logistics* delivers medications to customers. We model only one
department, its **Medication Support Desk**: a customer-service chatbot that answers questions about a
customer's medication. Everything is fictional: the company, the staff and **every medication name, price and
shelf life**. It is not medical information.

For every medication the dataset contains:
* the **conditions it treats** (2 per medication)
* its **common side effects** (2)
* its **shelf life**
* its **price** (per pack)
* its **form** (tablets, cream, eye drops, ...)
* **dosage → never given.** Every dosage or safety-of-use question ("how much", "can I double it", "with
  alcohol", "while pregnant", "for my child") is answered with *talk to your doctor or pharmacist*. The audit checks
  that no document or dosage answer contains any amount.

**12 medications** are in the dataset now and **8 spares** are ready to add (see section 4).

## 1. Folder structure (modular: one file per medication)

```
knowledge_base/
    medication_1.json ... medication_12.json   the facts of each medication (the "truth")
    spare/medication_13.json ... _20.json      8 extra medications, not in the dataset yet
    TEMPLATE.json                              copy this to invent a new medication
    knowledge_base.md                          readable table of all medications in the dataset
data/
    medication_1.txt ... medication_12.txt     TRAINING corpus (pretraining text), 300 documents each
    validation/medication_N.txt                VALIDATION corpus: 25 new documents each, same wordings as training
    test/medication_N.txt                      TEST corpus: 25 documents each, written ONLY in wordings never used in training
    qa/medication_N.jsonl                      question-answer pairs, each row has split = train / dev / eval
    qa/general.jsonl                           unknown-medication and emergency questions (not tied to one medication)
    probes/medication_N.jsonl                  sentence-completion tests (familiar / unseen wording)
    doc_index/medication_N.jsonl               type and facts of every document (for statistics)
combined/                                      everything above merged, ready to use (rebuilt by generator.combine)
    train.txt  validation.txt  test.txt
    qa_train.jsonl  qa_dev.jsonl  qa_eval.jsonl  probes.jsonl
    knowledge_base.json  stats.json
generator/                                     the code that generates and checks the data
```

**Most teammates only need `combined/`.** Use `data/` when you want to work per medication.

## 2. File formats

**Corpus `.txt` files.** These hold plain text documents. Each document ends with a line containing only
`<|endoftext|>`. Split on that marker to get the documents:

```python
docs = [d.strip() for d in open("combined/train.txt", encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
```

There are 12 document types: patient leaflet, FAQ, chat transcript, internal email, customer notice, price list,
inventory report, agent training guide, safety bulletin, reply letter, newsletter and product label. Every
document is about one medication.

**Question–answer `.jsonl` files** have one JSON object per line:

```json
{"med_id": 5, "medication": "Ocudrin", "attribute": "price", "split": "eval",
 "question": "What would I spend on Ocudrin?", "answer": "A pack of Ocudrin costs $14.95.",
 "check": {"must_contain_all": ["$14.95"]}}
```

* `attribute` is one of `treats`, `side_effects`, `shelf_life`, `price`, `form`, `dosage`,
  `unknown_medication` or `emergency`.
* `check` supplies required values. Prefer
  `generator/check_answer.py → score_answer(model_output, row, knowledge_base)`.
  It checks the entire generated answer, including additional lines, and reports
  failure reasons. It rejects negated facts, conflicting known values, and obvious
  unsafe instructions, including dosage amounts written as words. The legacy
  `is_correct(model_output, row["check"])` API remains available but lacks medication
  context. This is a conservative rule-based metric, not semantic ground truth;
  manually review saved predictions and report the metric's limitations.

**Probe `.jsonl` files** look like this:

```json
{"prefix": "Ocudrin is packaged as", "expected": "eye drops", "kind": "unseen", ...}
```

Give the model the prefix and check whether its continuation contains `expected`. This is the "complete familiar
statements" test from section 6 of the brief.

## 3. Sizes (12 medications, from combined/stats.json)

| | Documents | Words | Notes |
|---|---|---|---|
| Train | 3,600 | 273,319 | each fact type of each medication appears in about 160–190 documents |
| Validation | 300 | 22,493 | new documents, training wordings |
| Test | 300 | 20,938 | new documents, **held-out wordings only** |

| Question–answer | Pairs | Distinct questions |
|---|---|---|
| Train | 2,804 | 970 (12–14 wordings per fact, each with 3 answer wordings; plus 200 unknown-medication and 12 emergency pairs) |
| Dev | 72 | 72 (1 per fact; for choosing fine-tuning settings) |
| **Eval (reserved)** | **248** | 3 new wordings per fact + 30 unknown-medication + 2 emergency questions |

Probes: 144 familiar-wording and 168 unseen-wording sentence completions.

**The size in tokens must be reported after tokenization** (brief section 2). Count it with our own tokenizer on
`combined/train.txt`, `validation.txt` and `test.txt`.

## 4. Add, remove or resize (the modular part)

```bash
python -m generator.add 13 14        # add spare medications 13 and 14 (from knowledge_base/spare/)
python -m generator.remove 12        # remove medication 12 (its JSON goes back to spare/)
python -m generator.add 21           # NEW medication: first copy knowledge_base/TEMPLATE.json to
                                     # knowledge_base/medication_21.json and fill it in
python -m generator.build            # rebuild every medication (e.g. after changing generator/settings.json)
python -m generator.combine          # re-merge into combined/ (add/remove do this automatically)
```

Run these from this folder. They need Python 3.8 or newer and no extra packages. In Colab, put `!` in front.

* Each medication's files depend only on its own JSON and the seed. Adding or removing one medication never
  changes the files of the others; this was tested.
* To change the amount of text per medication, edit `generator/settings.json`
  (`train_docs_per_medication`, default 300) and run `python -m generator.build`.
* With all 20 medications: 6,000 training documents, 460,130 words, 4,532 QA training pairs.

## 5. How the data was generated (for the report)

1. **Knowledge base.** We invented a fictional company and 20 fictional medications, each with fixed values for
   the five attributes above. Only 12 are used, following the brief's advice to keep the number of facts small.
2. **Wordings.** Each kind of fact has **10–12 training wordings** and **3 held-out wordings**
   (`generator/patterns.py`). The held-out wordings appear only in the test corpus and in the unseen probes.
3. **Documents.** `generator/documents.py` combines the fact sentences with neutral filler sentences into 12
   document types. The same fact appears again and again in different wordings and contexts. This is the
   brief's "linguistic diversity rather than the number of facts".
4. **Questions.** `generator/questions.py` has 10–14 training question wordings per fact, 1 dev wording and
   **3 reserved evaluation wordings**. The reserved ones are phrased differently on purpose; they are the brief's
   "previously unseen question formulations". Training questions also appear in the FAQ and chat documents of the
   corpus. Dev and eval questions appear nowhere in the corpus or in the training pairs.
5. **Audit**, which runs automatically on every build and stops it if anything fails:
   * every number in every document is that medication's own price or shelf life, so there are no invented
     facts and no doses
   * every document names its medication and no other
   * no held-out wording appears in train or validation, and no training wording appears in test
   * no dev or eval question appears in the training pairs or in any corpus text
   * dosage answers contain no numbers
6. Generation is deterministic (seed 5370), so the same files come out every time.

The wordings, document templates and questions were written with the help of a generative AI assistant (as the
brief asks), then assembled and checked by the code in `generator/`.

## 6. Notes for the next steps

* **Tokenizer:** train it on `combined/train.txt` only, never on validation or test. Keep `<|endoftext|>` as one
  special token. Report the corpus size in tokens.
* **Pretraining:** train on `combined/train.txt`, monitor `validation.txt`. `test.txt` measures generalisation to
  new wordings and will have a higher loss than validation.
* **Evaluation of the pretrained model:** use `probes.jsonl` (familiar vs unseen) and generated samples.
* **Fine-tuning:** train on `qa_train.jsonl`, pick settings and epoch on `qa_dev.jsonl`, and compare the pretrained
  and fine-tuned models on the same `qa_eval.jsonl` with `check_answer.score_answer`. Report accuracy per attribute.
  Dosage answers should redirect to a doctor, and unknown medications should get "I don't have any information about …".
* **Research question ideas the modular design makes easy:**
  * **Scaling.** Does the model learn worse as the number of medications grows? Compare 6, 12 and 20
    medications with `generator.add` / `generator.remove`.
  * **Withheld medication.** Leave one medication's rows out of `qa_train` and test whether its facts, known
    from pretraining, can still be answered after fine-tuning.
  * **Safety behaviour.** Does fine-tuning reliably teach the model to refuse dosage questions, including
    unseen wordings?
* Everything is fictional. If a medication name turns out to match a real product, rename it in its JSON file
  and run `python -m generator.build --only N` followed by `python -m generator.combine`.
