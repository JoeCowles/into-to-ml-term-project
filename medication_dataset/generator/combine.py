"""
Merge the files of every medication currently in data/ into ready-to-use files in combined/.
Run this after adding or removing medications.

    python -m generator.combine

combined/
    train.txt, validation.txt, test.txt     corpus splits; documents separated by a line "<|endoftext|>"
    qa_train.jsonl, qa_dev.jsonl, qa_eval.jsonl   question-answer pairs (medication + general questions)
    probes.jsonl                            sentence-completion probes
    knowledge_base.json                     facts of the medications included
    stats.json                              sizes, document types, fact mentions
knowledge_base/knowledge_base.md is rewritten as a readable table of the medications included.
"""
import collections
import glob
import json
import os
import re

from .build import DATA, KB_DIR, ROOT, SEP, load_kb, read_txt
from .documents import fields

OUT = os.path.join(ROOT, "combined")


def med_ids():
    ids = [int(re.search(r"medication_(\d+)\.txt$", p).group(1)) for p in glob.glob(os.path.join(DATA, "medication_*.txt"))]
    return sorted(ids)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = med_ids()
    kb = {m["id"]: m for m in load_kb()}
    missing_kb = [i for i in ids if i not in kb]
    assert not missing_kb, f"data files exist for medications {missing_kb} but their knowledge-base file is missing"

    stats = {"medications": [f"medication_{i} ({kb[i]['name']})" for i in ids], "splits": {}}
    for split, sub in (("train", ""), ("validation", "validation"), ("test", "test")):
        docs = []
        for i in ids:
            docs += read_txt(os.path.join(DATA, sub, f"medication_{i}.txt"))
        with open(os.path.join(OUT, f"{split}.txt"), "w", encoding="utf-8") as f:
            for d in docs:
                f.write(d + "\n" + SEP + "\n")
        stats["splits"][split] = {"documents": len(docs), "words": sum(len(d.split()) for d in docs),
                                  "characters": sum(len(d) for d in docs)}

    qa = [r for i in ids for r in read_jsonl(os.path.join(DATA, "qa", f"medication_{i}.jsonl"))]
    qa += read_jsonl(os.path.join(DATA, "qa", "general.jsonl"))
    for split in ("train", "dev", "eval"):
        rows = [r for r in qa if r["split"] == split]
        with open(os.path.join(OUT, f"qa_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"qa_{split}"] = {"pairs": len(rows), "distinct_questions": len({r["question"] for r in rows}),
                                "by_attribute": dict(collections.Counter(r["attribute"] for r in rows))}

    probes = [r for i in ids for r in read_jsonl(os.path.join(DATA, "probes", f"medication_{i}.jsonl"))]
    with open(os.path.join(OUT, "probes.jsonl"), "w", encoding="utf-8") as f:
        for r in probes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["probes"] = dict(collections.Counter(r["kind"] for r in probes))

    index = [r for i in ids for r in read_jsonl(os.path.join(DATA, "doc_index", f"medication_{i}.jsonl"))]
    stats["document_types_train"] = dict(collections.Counter(r["type"] for r in index if r["split"] == "train").most_common())
    stats["document_types_test"] = dict(collections.Counter(r["type"] for r in index if r["split"] == "test").most_common())
    mentions = collections.Counter(a for r in index if r["split"] == "train" for a in r["facts"])
    stats["documents_mentioning_each_fact_type_per_medication_train"] = {k: round(v / len(ids), 1) for k, v in mentions.items()}
    stats["note"] = "Token counts depend on the tokenizer; count them after tokenization (brief section 2)."

    json.dump([kb[i] for i in ids], open(os.path.join(OUT, "knowledge_base.json"), "w"), indent=2)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)

    lines = ["# Knowledge base: Halverin Medical Logistics, Medication Support Desk", "",
             "Everything here is fictional (company, staff, medication names, prices, shelf lives). The medications",
             "and their facts were invented for a class project and must not be used as medical information.", "",
             "Dosage rule: the desk never states a dose. Every dosage or safety-of-use question is referred to a",
             "doctor or pharmacist; overdose or a serious reaction -> contact a doctor or emergency services.", "",
             f"Medications in the dataset: {len(ids)}", "",
             "| file | medication | form | treats | common side effects | shelf life | price per pack |",
             "|---|---|---|---|---|---|---|"]
    for i in ids:
        f = fields(kb[i])
        lines.append(f"| medication_{i} | {f['name']} | {f['form']} | {f['cond1']}, {f['cond2']} | {f['se1']}, {f['se2']} "
                     f"| {f['shelf']} | {f['price']} |")
    spare = load_kb(include_spare=True)
    spare = [m for m in spare if m["id"] not in ids]
    if spare:
        lines += ["", "Spare medications (in knowledge_base/spare/, not in the dataset; add with `python -m generator.add N`): "
                  + ", ".join(f"{m['id']} {m['name']}" for m in spare)]
    open(os.path.join(KB_DIR, "knowledge_base.md"), "w").write("\n".join(lines) + "\n")

    s = stats["splits"]
    print(f"{len(ids)} medications | train {s['train']['documents']} docs / {s['train']['words']:,} words | "
          f"validation {s['validation']['documents']} | test {s['test']['documents']} | "
          f"QA train {stats['qa_train']['pairs']} pairs, dev {stats['qa_dev']['pairs']}, eval {stats['qa_eval']['pairs']} "
          f"| probes {len(probes)}  -> combined/")


if __name__ == "__main__":
    main()
