"""
Language utilities: word frequencies, prefix lookup, edit distance, normalization.

Designed to be swappable with a larger lexicon or external LM later.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterator


def normalize_fingerspell_letters(s: str) -> str:
    """
    Map common vision / OCR confusions before lexical decode.

    Classifiers often emit digit ``1`` when the handshape is ``I``; map those so edit
    distance to dictionary words is not blind to the intended letter.
    """
    out: list[str] = []
    for c in s:
        if c == "1":
            out.append("I")
        elif c.isalpha():
            out.append(c.upper())
        elif c.isspace():
            out.append(" ")
    return " ".join("".join(out).split())


def normalize_letters(s: str) -> str:
    """Keep A–Z and spaces; uppercase; applies fingerspell noise rules (e.g. 1→I)."""
    return normalize_fingerspell_letters(s)


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance (insert/delete/substitute)."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = range(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


class WordFrequencyModel:
    """
    Simple unigram model from a TSV or whitespace file:

        word<TAB>count
        hello 1000

    Words are lowercased for lookup; output casing is handled by callers.
    """

    def __init__(self, path: Path | None = None):
        self._freq: dict[str, float] = {}
        self._max_freq: float = 1.0
        if path is not None and path.is_file():
            self.load(path)

    def load(self, path: Path) -> None:
        self._freq.clear()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\t\s]+", line, maxsplit=1)
            w = parts[0].lower()
            if len(parts) > 1:
                try:
                    c = float(parts[1])
                except ValueError:
                    c = 1.0
            else:
                c = 1.0
            self._freq[w] = self._freq.get(w, 0.0) + c
        self._max_freq = max(self._freq.values(), default=1.0)

    def frequency(self, word: str) -> float:
        return self._freq.get(word.lower(), 0.0)

    def log_freq(self, word: str) -> float:
        """log(1 + freq), bounded."""
        return math.log1p(self.frequency(word))

    def normalized_log_freq(self, word: str) -> float:
        """Roughly [0, 1] for scoring."""
        if not self._freq:
            return 0.0
        return math.log1p(self.frequency(word)) / math.log1p(self._max_freq + 1e-9)

    def words_with_prefix(self, prefix: str) -> Iterator[str]:
        p = prefix.lower()
        for w in self._freq:
            if w.startswith(p):
                yield w

    def all_words(self) -> Iterator[str]:
        return iter(sorted(self._freq.keys()))

    def __len__(self) -> int:
        return len(self._freq)


class BigramModel:
    """Sparse bigram counts: lines like ``hello world<TAB>10`` or ``hello world 10``."""

    def __init__(self, path: Path | None = None):
        self._counts: dict[tuple[str, str], float] = {}
        self._totals: dict[str, float] = {}
        if path is not None and path.is_file():
            self.load(path)

    def load(self, path: Path) -> None:
        self._counts.clear()
        self._totals.clear()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                left, _, right = line.partition("\t")
                parts = left.split()
                try:
                    c = float(right.strip())
                except ValueError:
                    c = 1.0
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                c = float(parts[-1]) if len(parts) > 2 else 1.0
                parts = parts[:-1] if len(parts) > 2 else parts
            if len(parts) < 2:
                continue
            w1, w2 = parts[0].lower(), parts[1].lower()
            self._counts[(w1, w2)] = self._counts.get((w1, w2), 0.0) + c
            self._totals[w1] = self._totals.get(w1, 0.0) + c

    def __bool__(self) -> bool:
        return bool(self._counts)

    def score(self, prev: str, nxt: str) -> float:
        """Conditional log-prob style score in [0, ~1], or 0 if unknown."""
        p, n = prev.lower(), nxt.lower()
        c = self._counts.get((p, n), 0.0)
        if c <= 0:
            return 0.0
        tot = self._totals.get(p, c)
        p_cond = c / tot
        return math.log1p(p_cond) / math.log(2.0)  # squash to friendly range
