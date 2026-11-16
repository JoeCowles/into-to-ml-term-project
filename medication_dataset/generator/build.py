"""
Build the data files of one, several or all medications.

    python -m generator.build                 # every medication that has a file in knowledge_base/
    python -m generator.build --only 3 7      # just medications 3 and 7 (others are left untouched)

Each medication N gets its own files:
    data/medication_N.txt              training corpus (pretraining text)
    data/validation/medication_N.txt   validation corpus: new documents, training wordings
    data/test/medication_N.txt         test corpus: new documents, ONLY held-out wordings
    data/qa/medication_N.jsonl         question-answer pairs (split = train / dev / eval)
    data/probes/medication_N.jsonl     sentence-completion probes (familiar / unseen wording)
A medication's text depends only on its own knowledge-base file and the seed, so building or deleting one
medication never changes the files of another.
Documents inside a .txt file are separated by a line containing only <|endoftext|>.
"""
import argparse
import glob
import json
import os
import random
import re

from .documents import fields, fill, make_doc
from .patterns import FACT_ATTRS, HELDOUT, TRAIN
from .questions import (A, EMERGENCY_A, EMERGENCY_EVAL, EMERGENCY_TRAIN, Q, UNKNOWN_A, UNKNOWN_NAMES_EVAL,
                        UNKNOWN_NAMES_TRAIN, UNKNOWN_Q_EVAL, UNKNOWN_Q_TRAIN)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR, DATA = os.path.join(ROOT, "knowledge_base"), os.path.join(ROOT, "data")
SEP = "<|endoftext|>"
SETTINGS = json.load(open(os.path.join(ROOT, "generator", "settings.json")))


def load_kb(include_spare=False):
    files = glob.glob(os.path.join(KB_DIR, "medication_*.json"))
    if include_spare:
        files += glob.glob(os.path.join(KB_DIR, "spare", "medication_*.json"))
    meds = [json.load(open(f)) for f in files]
    return sorted(meds, key=lambda m: m["id"])


def write_txt(path, docs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d["text"].strip() + "\n" + SEP + "\n")


def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return [d.strip() for d in f.read().split(SEP) if d.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_for(attr, f):
    """What a correct answer must contain (used by generator/check_answer.py)."""
    return {"treats": {"must_contain_all": [f["cond1"], f["cond2"]]},
            "side_effects": {"must_contain_all": [f["se1"], f["se2"]]},
            "shelf_life": {"must_contain_all": [f["shelf"]]},
            "price": {"must_contain_all": [f["price"]]},
            "form": {"must_contain_all": [f["form"]]},
            "dosage": {"must_contain_any": ["doctor", "pharmacist"], "must_not_contain_digits": True}}[attr]


def make_docs(m, rng, n, bank, test, seen):
    docs, tries = [], 0
    while len(docs) < n and tries < n * 50:
        tries += 1
        d = make_doc(m, rng, bank, test)
        if d["text"] not in seen:
            seen.add(d["text"])
            docs.append(d)
    return docs


def make_qa(m):
    f, rows = fields(m), []
    base = {"med_id": m["id"], "medication": m["name"]}
    for attr in FACT_ATTRS:
        chk = check_for(attr, f)
        for q in Q[attr]["train"]:
            for a in A[attr]:
                rows.append({**base, "attribute": attr, "split": "train", "question": fill(q, f),
                             "answer": fill(a, f), "check": chk})
        for split in ("dev", "eval"):
            for q in Q[attr][split]:
                rows.append({**base, "attribute": attr, "split": split, "question": fill(q, f),
                             "answer": fill(A[attr][0], f), "check": chk})
    return rows


def make_general_qa():
    rows = []
    for split, names, qs in (("train", UNKNOWN_NAMES_TRAIN, UNKNOWN_Q_TRAIN), ("eval", UNKNOWN_NAMES_EVAL, UNKNOWN_Q_EVAL)):
        for name in names:
            for q in qs:
                answers = UNKNOWN_A if split == "train" else UNKNOWN_A[:1]
                for a in answers:
                    rows.append({"med_id": None, "medication": name, "attribute": "unknown_medication", "split": split,
                                 "question": q.format(name=name), "answer": a.format(name=name),
                                 "check": {"must_contain_any": ["don't have", "not a medication I have"]}})
    for split, qs in (("train", EMERGENCY_TRAIN), ("eval", EMERGENCY_EVAL)):
        for q in qs:
            for a in (EMERGENCY_A if split == "train" else EMERGENCY_A[:1]):
                rows.append({"med_id": None, "medication": None, "attribute": "emergency", "split": split,
                             "question": q, "answer": a, "check": {"must_contain_any": ["emergency"]}})
    return rows


VALUE_KEYS = {"treats": "{cond1}", "side_effects": "{se1}", "shelf_life": "{shelf}", "price": "{price}",
              "form": "{forms}", "dosage": "doctor"}


def make_probes(m, rng):
    """Sentence completion: give the start of a fact sentence, the model should continue with the value."""
    f, rows = fields(m), []
    expected = {"treats": f["cond1"], "side_effects": f["se1"], "shelf_life": f["shelf"], "price": f["price"],
                "form": f["form"], "dosage": "doctor"}
    for kind, bank in (("familiar", TRAIN), ("unseen", HELDOUT)):
        for attr in FACT_ATTRS:
            ok = [p for p in bank[attr] if VALUE_KEYS[attr] in p and "{name}" in p[:p.index(VALUE_KEYS[attr])]]
            for p in (rng.sample(ok, min(2, len(ok))) if kind == "familiar" else ok):
                full = fill(p, f)
                cut = full.index(expected[attr] if attr != "form" else f["forms"])
                rows.append({"med_id": m["id"], "medication": m["name"], "attribute": attr, "kind": kind,
                             "prefix": full[:cut].rstrip(), "expected": expected[attr], "full_sentence": full})
    return rows


def audit(m, all_meds, train, val, test, qa):
    """Raises AssertionError if anything in this medication's files is inconsistent."""
    f = fields(m)
    allowed_numbers = {f"{m['price_usd']:.2f}", str(m["shelf_life_months"])}
    others = [o["name"] for o in all_meds if o["id"] != m["id"]]
    for split, docs in (("train", train), ("validation", val), ("test", test)):
        for d in docs:
            t = d["text"]
            assert m["name"] in t, f"{split}: document without the medication name"
            nums = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?m)^\d\. ", "", t)))  # leaflet headings "1. " are not facts
            assert nums <= allowed_numbers, f"{split}: unexpected number(s) {nums - allowed_numbers} in {t[:80]}"
            assert not any(o in t for o in others), f"{split}: mentions another medication"
    train_text = "\n".join(d["text"] for d in train + val)
    test_text = "\n".join(d["text"] for d in test)
    for attr in list(HELDOUT):
        for p in HELDOUT[attr]:
            assert fill(p, f) not in train_text, f"held-out wording leaked into train/val: {p}"
    for attr in FACT_ATTRS:
        for p in TRAIN[attr]:
            assert fill(p, f) not in test_text, f"training wording found in test: {p}"
    train_q = {r["question"].lower() for r in qa if r["split"] == "train"}
    for r in qa:
        if r["split"] in ("dev", "eval"):
            assert r["question"].lower() not in train_q, f"{r['split']} question also in train: {r['question']}"
            assert r["question"] not in train_text + test_text, f"{r['split']} question appears in the corpus"
        if r["attribute"] == "dosage":
            assert not re.search(r"\d", r["answer"])
    return True


def build_one(m, all_meds):
    s = SETTINGS
    rng = random.Random(s["seed"] * 1000 + m["id"])
    seen = set()
    train = make_docs(m, rng, s["train_docs_per_medication"], TRAIN, False, seen)
    val = make_docs(m, rng, s["validation_docs_per_medication"], TRAIN, False, seen)
    test = make_docs(m, rng, s["test_docs_per_medication"], HELDOUT, True, seen)
    qa = make_qa(m)
    probes = make_probes(m, rng)
    audit(m, all_meds, train, val, test, qa)
    n = m["id"]
    write_txt(os.path.join(DATA, f"medication_{n}.txt"), train)
    write_txt(os.path.join(DATA, "validation", f"medication_{n}.txt"), val)
    write_txt(os.path.join(DATA, "test", f"medication_{n}.txt"), test)
    write_jsonl(os.path.join(DATA, "qa", f"medication_{n}.jsonl"), qa)
    write_jsonl(os.path.join(DATA, "probes", f"medication_{n}.jsonl"), probes)
    write_jsonl(os.path.join(DATA, "doc_index", f"medication_{n}.jsonl"),
                [{"split": sp, "index": i, "type": d["type"], "facts": d["facts"]}
                 for sp, docs in (("train", train), ("validation", val), ("test", test)) for i, d in enumerate(docs)])
    print(f"medication_{n} ({m['name']}): {len(train)} train, {len(val)} validation, {len(test)} test documents, "
          f"{sum(r['split'] == 'train' for r in qa)} QA train pairs, audit passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="medication ids to (re)build")
    a = ap.parse_args()
    all_meds = load_kb(include_spare=True)
    active = load_kb()
    todo = [m for m in active if not a.only or m["id"] in a.only]
    if a.only:
        missing = set(a.only) - {m["id"] for m in todo}
        assert not missing, f"no knowledge_base/medication_N.json for {missing} (spare ones: use generator.add)"
    for m in todo:
        build_one(m, all_meds)
    write_jsonl(os.path.join(DATA, "qa", "general.jsonl"), make_general_qa())


if __name__ == "__main__":
    main()
