"""
Map noisy letter strings to dictionary words using frequency and edit distance.

Works on a single word at a time (no spaces inside ``noisy_letters``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DecoderConfig
from .language_utils import WordFrequencyModel, levenshtein, normalize_letters


@dataclass
class WordCandidate:
    """One lexical hypothesis with a score (higher = better)."""

    word: str
    score: float
    edit_distance: int
    avg_letter_confidence: float
    details: str = ""


def _score_word(
    candidate: str,
    noisy: str,
    dist: int,
    avg_conf: float,
    lex: WordFrequencyModel,
    config: DecoderConfig,
) -> float:
    nf = normalize_letters(noisy).replace(" ", "")
    cw = candidate.lower()
    score = config.weight_visual_confidence * avg_conf
    score += config.weight_word_frequency * lex.normalized_log_freq(cw)
    score -= config.weight_edit_penalty * dist
    # Length compatibility
    score -= config.weight_length_prior * abs(len(candidate) - len(nf))
    # Prefix bonus: candidate extends a prefix that matches noisy
    if nf and cw.startswith(nf[: min(3, len(nf))].lower()):
        score += config.weight_prefix_bonus
    return score


def decode_word(
    noisy_letters: str,
    lex: WordFrequencyModel,
    config: DecoderConfig,
    per_letter_confidence: list[float] | None = None,
) -> list[WordCandidate]:
    """
    Return ranked ``WordCandidate`` list for one noisy word (letters only).

    ``per_letter_confidence`` should align with ``noisy_letters`` letters (same length);
    if omitted, uses 1.0 per letter.
    """
    nf = normalize_letters(noisy_letters).replace(" ", "")
    if not nf:
        return []

    if per_letter_confidence is not None and len(per_letter_confidence) == len(nf):
        avg_conf = sum(per_letter_confidence) / len(nf)
    else:
        avg_conf = 1.0

    candidates: list[WordCandidate] = []
    seen: set[str] = set()

    for w in lex.all_words():
        if abs(len(w) - len(nf)) > max(3, config.max_edit_distance + 2):
            continue
        d = levenshtein(nf, w.upper())
        if d > config.max_edit_distance:
            continue
        sc = _score_word(w, nf, d, avg_conf, lex, config)
        candidates.append(
            WordCandidate(
                word=w,
                score=sc,
                edit_distance=d,
                avg_letter_confidence=avg_conf,
                details=f"edit={d},freq~{lex.normalized_log_freq(w):.3f}",
            )
        )
        seen.add(w.lower())

    # Prefix / autocomplete: dictionary words whose prefix matches the first few letters of ``nf``
    pref = nf[: max(1, min(4, len(nf)))].lower()
    for w in lex.words_with_prefix(pref):
        wl = w.lower()
        if wl in seen:
            continue
        d = levenshtein(nf, w.upper())
        if d > config.max_edit_distance + 1:
            continue
        sc = _score_word(w, nf, d, avg_conf, lex, config) + 0.05
        candidates.append(
            WordCandidate(word=w, score=sc, edit_distance=d, avg_letter_confidence=avg_conf, details="prefix")
        )
        seen.add(wl)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: config.max_word_candidates]


def load_default_lexicon() -> WordFrequencyModel:
    """Load packaged sample vocabulary next to this package."""
    here = Path(__file__).resolve().parent / "data" / "word_freq_sample.txt"
    return WordFrequencyModel(here)
