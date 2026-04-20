"""
Assemble decoded words into sentences and apply optional bigram smoothing.
"""

from __future__ import annotations

from pathlib import Path

from .beam_search import BeamHypothesis, beam_search_word_lists
from .config import DecoderConfig
from .language_utils import BigramModel
from .word_decoder import WordCandidate, decode_word


def split_smoothed_into_words(smoothed_letters: str) -> list[str]:
    """Split on whitespace (word boundaries from temporal pauses)."""
    return [w for w in smoothed_letters.upper().split() if w]


def decode_sentence_words(
    word_fragments: list[str],
    lex,
    config: DecoderConfig,
    bigram: BigramModel | None = None,
) -> tuple[str, list[BeamHypothesis], list[list[WordCandidate]]]:
    """
    Decode each fragment (noisy letters) to a word; combine with beam search.

    Returns (best_sentence_string, top_beam_hypotheses, word_lattice).
    """
    lattice: list[list[WordCandidate]] = []
    for frag in word_fragments:
        cands = decode_word(frag, lex, config)
        if not cands:
            lattice.append(
                [
                    WordCandidate(
                        word=frag.lower(),
                        score=-1.0,
                        edit_distance=99,
                        avg_letter_confidence=0.0,
                        details="OOV",
                    )
                ]
            )
        else:
            lattice.append(cands)

    def bi(prev: str, nxt: str) -> float:
        if bigram is None:
            return 0.0
        return bigram.score(prev, nxt)

    hyps = beam_search_word_lists(lattice, config, bigram_score=bi if bigram else None)
    if not hyps:
        return "", [], lattice
    best = " ".join(hyps[0].words)
    return best, hyps, lattice


def load_optional_bigrams(path: Path | None) -> BigramModel | None:
    if path is None or not path.is_file():
        return None
    m = BigramModel(path)
    return m if m else None
