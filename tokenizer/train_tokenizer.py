"""
Build the character-level (one-hot) tokenizer and report corpus sizes in tokens.

The vocabulary is fixed (newline + printable ASCII + 3 special tokens), so nothing
is learned from validation or test data. The script checks that train.txt is fully
covered and reports how many characters in each split fall outside the vocabulary.

Run from the repository root:
    python -m tokenizer.train_tokenizer

Outputs (in tokenizer/):
    tokenizer.json       the tokenizer (load with CharTokenizer.load)
    vocab.txt            every token ID and its character
    token_stats.json     corpus sizes in tokens and document-length statistics
"""

import argparse
import json
import statistics
from pathlib import Path

from tokenizer.char_tokenizer import CharTokenizer

EOT = "<|endoftext|>"


def read(path):
    return Path(path).read_text(encoding="utf-8")


def split_docs(text):
    return [d.strip() for d in text.split(EOT) if d.strip()]


def qa_text(path):
    rows = [json.loads(l) for l in read(path).splitlines() if l.strip()]
    return [r["question"] + "\n" + r["answer"] for r in rows]


def length_stats(lengths):
    lengths = sorted(lengths)
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    return {"mean": round(statistics.mean(lengths), 1), "median": pct(0.5),
            "p95": pct(0.95), "max": lengths[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="medication_dataset/combined", help="folder with train/validation/test files")
    ap.add_argument("--out", default="tokenizer", help="where to save the tokenizer files")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    tok = CharTokenizer()
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(tok.itos):
            f.write(f"{i}\t{s!r}\n")

    splits = {name: read(data / f"{name}.txt") for name in ("train", "validation", "test")}

    # ---- checks ----
    assert tok.encode(EOT) == [tok.eot_id], "<|endoftext|> must be a single token"
    for name, text in splits.items():
        ids = tok.encode(text)
        assert tok.unk_id not in ids, f"{name} contains characters outside the vocabulary"
        assert tok.decode(ids) == text, f"round-trip failed on {name}"
    assert tok.decode(tok.encode("Is Zorbanex safe? $99.99")) == "Is Zorbanex safe? $99.99"
    assert tok.encode("café") [-1] == tok.unk_id          # non-ASCII -> <|unk|>
    assert CharTokenizer.load(out / "tokenizer.json").itos == tok.itos

    # ---- statistics ----
    train_chars_used = sorted(set(splits["train"].replace(EOT, "")))
    stats = {"tokenizer": "character-level, one-hot (fixed ASCII vocabulary)",
             "vocab_size": tok.vocab_size,
             "special_tokens": {s: tok.stoi[s] for s in tok.special_tokens},
             "characters_used_in_train": len(train_chars_used),
             "corpus": {}, "qa": {}}
    for name, text in splits.items():
        docs = split_docs(text)
        stats["corpus"][name] = {
            "documents": len(docs),
            "tokens": len(tok.encode(text)),
            "tokens_per_document": length_stats([len(tok.encode(d)) + 1 for d in docs]),
        }
    for name in ("qa_train", "qa_dev", "qa_eval"):
        pairs = qa_text(data / f"{name}.jsonl")
        lengths = [len(tok.encode(p)) for p in pairs]
        unk = sum(tok.encode(p).count(tok.unk_id) for p in pairs)
        stats["qa"][name] = {"pairs": len(pairs), "tokens": sum(lengths),
                             "max_pair_tokens": max(lengths), "unknown_characters": unk}
    (out / "token_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- summary ----
    print(f"Saved {out / 'tokenizer.json'}, {out / 'vocab.txt'}, {out / 'token_stats.json'}")
    print(f"Vocabulary size: {tok.vocab_size} (96 characters + special tokens "
          f"{stats['special_tokens']}); train.txt uses {len(train_chars_used)} of them")
    print(f"\n{'split':<11}{'docs':>6}{'tokens':>11}{'mean len':>10}{'p95 len':>9}{'max len':>9}")
    for name, s in stats["corpus"].items():
        d = s["tokens_per_document"]
        print(f"{name:<11}{s['documents']:>6}{s['tokens']:>11,}{d['mean']:>10}{d['p95']:>9}{d['max']:>9}")
    for name, s in stats["qa"].items():
        print(f"{name:<11}{s['pairs']:>6} pairs, {s['tokens']:,} tokens, longest pair "
              f"{s['max_pair_tokens']}, unknown chars {s['unknown_characters']}")

    example = "Allervane costs $12.40."
    ids = tok.encode(example)
    print(f"\nExample: {example!r}\nIDs:     {ids}")
    try:
        print(f"One-hot shape: {tuple(tok.one_hot(ids).shape)}")
    except ImportError:
        print("(install torch to use tok.one_hot)")
    print("\nAll checks passed (round-trip on all splits, no unknown characters, "
          "<|endoftext|> = 1 token, save/load).")


if __name__ == "__main__":
    main()
