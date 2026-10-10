"""
EPM NLP Preprocessing & Vocabulary
==================================

Pure Python & NumPy utilities for text cleaning, regex tokenization,
and vocabulary indexing with special tokens (<pad>, <unk>).

MATHEMATICAL & ALGORITHMIC FOUNDATION
-------------------------------------
1. Text Normalization:
   Transforms unstructured raw characters into canonical tokens:
     - Case folding: lowercase to merge "Vibration" and "vibration"
     - Punctuation separation or removal
     - Normalizing whitespace

2. Vocabulary Mapping:
   Bijective mapping between discrete strings and continuous integers:
     token -> index: V: T -> {0, 1, ..., |V|-1}
     index -> token: V^(-1): {0, 1, ..., |V|-1} -> T
   
   Special Tokens:
     <pad> (id 0): Used to align variable-length sequences to fixed tensor length L
     <unk> (id 1): Replaces out-of-vocabulary (OOV) words seen during inference
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence
import numpy as np


def clean_text(text: str) -> str:
    """Normalize raw text: lowercase, strip special characters, collapse whitespace."""
    text = text.lower()
    # Replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> list[str]:
    """Tokenize cleaned text by splitting on whitespace and word boundaries."""
    cleaned = clean_text(text)
    return [t for t in cleaned.split(" ") if t]


class Vocabulary:
    """
    Vocabulary builder that maps tokens to unique integer IDs and vice versa.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of tokens to keep in vocabulary.
    min_freq : int, default=1
        Minimum frequency for a word to be included in vocabulary.
    pad_token : str, default="<pad>"
        Padding token for sequence alignment.
    unk_token : str, default="<unk>"
        Out-of-vocabulary token for unseen words.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ) -> None:
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.token_counts: Counter[str] = Counter()

        # Reserve indices for special tokens
        self.pad_id = self._add_token(self.pad_token)
        self.unk_id = self._add_token(self.unk_token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            new_id = len(self.token2id)
            self.token2id[token] = new_id
            self.id2token[new_id] = token
            return new_id
        return self.token2id[token]

    def build_vocab(self, corpus: Sequence[str]) -> Vocabulary:
        """Count frequencies across corpus and build token dictionary."""
        self.token_counts = Counter()
        for doc in corpus:
            tokens = simple_tokenize(doc)
            self.token_counts.update(tokens)

        # Sort by frequency descending
        sorted_tokens = [
            token
            for token, freq in self.token_counts.most_common()
            if freq >= self.min_freq and token not in (self.pad_token, self.unk_token)
        ]

        if self.max_size is not None:
            # Leave room for pad and unk
            sorted_tokens = sorted_tokens[: max(0, self.max_size - 2)]

        for token in sorted_tokens:
            self._add_token(token)

        return self

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, text: str, max_length: Optional[int] = None) -> list[int]:
        """Convert a sentence into a list of token IDs, optionally padded/truncated."""
        tokens = simple_tokenize(text)
        token_ids = [self.token2id.get(t, self.unk_id) for t in tokens]

        if max_length is not None:
            if len(token_ids) < max_length:
                # Pad
                token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))
            else:
                # Truncate
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special: bool = False) -> str:
        """Convert a sequence of token IDs back into text."""
        tokens = []
        for tid in token_ids:
            t = self.id2token.get(tid, self.unk_token)
            if skip_special and t in (self.pad_token, self.unk_token):
                continue
            tokens.append(t)
        return " ".join(tokens)
