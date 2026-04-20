"""
Beam search over word candidates for multi-word / multi-hypothesis decoding.

Keeps several word-level hypotheses with combined scores (visual + lexicon + optional bigram).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import DecoderConfig
from .word_decoder import WordCandidate


@dataclass
class BeamHypothesis:
    """A partial or full decode: sequence of chosen words and cumulative score."""

    words: list[str]
    score: float


def beam_search_word_lists(
    word_lattice: list[list[WordCandidate]],
    config: DecoderConfig,
    bigram_score: Callable[[str, str], float] | None = None,
) -> list[BeamHypothesis]:
    """
    ``word_lattice[i]`` = ranked candidates for word position ``i`` (non-empty lists).

    Combines local word scores; optionally adds ``bigram_score(prev, curr)`` when
    ``bigram_score`` is provided.
    """
    if not word_lattice:
        return [BeamHypothesis(words=[], score=0.0)]

    beams: list[BeamHypothesis] = [BeamHypothesis(words=[], score=0.0)]

    for position, cands in enumerate(word_lattice):
        if not cands:
            continue
        next_beams: list[BeamHypothesis] = []
        for hyp in beams:
            for wc in cands[: config.beam_width]:
                add = wc.score
                if bigram_score and hyp.words:
                    add += config.weight_bigram * bigram_score(hyp.words[-1], wc.word)
                next_beams.append(
                    BeamHypothesis(words=hyp.words + [wc.word], score=hyp.score + add)
                )
        next_beams.sort(key=lambda h: h.score, reverse=True)
        beams = next_beams[: config.beam_width]

    return beams
