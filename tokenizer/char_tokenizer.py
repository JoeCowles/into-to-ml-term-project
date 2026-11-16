"""
Character-level tokenizer with one-hot encoding.

Vocabulary
----------
Every printable ASCII character (space through '~', 95 characters) plus newline
gets its own ID, followed by three special tokens:
    <|endoftext|>  document separator (one token, never split)
    <|pad|>        padding for batches during fine-tuning
    <|unk|>        any character outside the vocabulary (e.g. non-ASCII)
Total: 96 + 3 = 99 tokens.

Each token ID is represented as a one-hot vector of length vocab_size.
In the model, nn.Embedding(vocab_size, d_model) does exactly
"one-hot vector x embedding matrix" (it just looks up the row directly).

Usage
-----
    from tokenizer.char_tokenizer import CharTokenizer
    tok = CharTokenizer.load("tokenizer/tokenizer.json")
    ids = tok.encode("What does Allervane cost?")
    text = tok.decode(ids)
    vecs = tok.one_hot(ids)     # torch tensor, shape (len(ids), vocab_size)
    tok.vocab_size              # 99 -> rows of the embedding matrix / outputs of the LM head
"""

import json
import re

BASE_CHARS = ["\n"] + [chr(c) for c in range(32, 127)]          # newline + printable ASCII
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


class CharTokenizer:
    def __init__(self, chars=BASE_CHARS, special_tokens=DEFAULT_SPECIAL_TOKENS):
        self.chars = list(chars)
        self.special_tokens = list(special_tokens)
        self.itos = self.chars + self.special_tokens               # id -> string
        self.stoi = {s: i for i, s in enumerate(self.itos)}        # string -> id
        self.unk_id = self.stoi["<|unk|>"]
        self._special_regex = re.compile(
            "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")

    @property
    def vocab_size(self):
        return len(self.itos)

    @property
    def eot_id(self):
        return self.stoi["<|endoftext|>"]

    @property
    def pad_id(self):
        return self.stoi["<|pad|>"]

    def encode(self, text):
        """Text -> list of IDs. Special tokens in the text become single IDs."""
        ids = []
        for part in self._special_regex.split(text):
            if part in self.special_tokens:
                ids.append(self.stoi[part])
            else:
                ids.extend(self.stoi.get(ch, self.unk_id) for ch in part)
        return ids

    def decode(self, ids):
        """List of IDs -> text."""
        return "".join(self.itos[i] for i in ids)

    def one_hot(self, ids):
        """List of IDs -> one-hot tensor of shape (len(ids), vocab_size)."""
        import torch
        import torch.nn.functional as F
        return F.one_hot(torch.tensor(ids, dtype=torch.long), self.vocab_size).float()

    def save(self, path):
        data = {"type": "character-level (one-hot)", "vocab_size": self.vocab_size,
                "chars": self.chars, "special_tokens": self.special_tokens}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["chars"], data["special_tokens"])
        assert tok.vocab_size == data["vocab_size"]
        return tok
