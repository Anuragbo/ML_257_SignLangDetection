"""
High-level decoder: vision outputs → smoothed letters → words → sentence.

This module imports nothing from ``part1_letter_classifier`` or any specific model.
Swap CNN / YOLO / MobileNet by producing the same ``FramePrediction`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_PKG_DATA = Path(__file__).resolve().parent / "data"
_DEFAULT_BIGRAMS = _PKG_DATA / "bigrams_sample.txt"

from .beam_search import BeamHypothesis
from .config import DecoderConfig
from .language_utils import WordFrequencyModel, normalize_fingerspell_letters
from .sentence_builder import decode_sentence_words, load_optional_bigrams, split_smoothed_into_words
from .temporal_smoothing import FramePrediction, SmoothedOutput, smooth_frames, smooth_frames_video
from .word_decoder import WordCandidate, decode_word, load_default_lexicon


@dataclass
class PipelineResult:
    """All useful outputs for UI, logging, or evaluation."""

    raw_smoothed: SmoothedOutput
    """Letters + pause spaces after temporal smoothing."""

    letters_normalized: str
    """Smoothed string after fingerspell noise rules (e.g. 1→I); used for split + decode."""

    word_fragments: list[str]
    """Letter chunks split on long blanks (one noisy “word” each)."""

    sentence: str
    """Best full-sentence string after beam search."""

    beam_hypotheses: list[BeamHypothesis]
    """Top sentence-level hypotheses with cumulative scores."""

    word_lattice: list[list[WordCandidate]]
    """Per-word ranked candidates (for debugging / UI alternatives)."""

    cleaned_letters_no_spaces: str
    """Smoothed letters A–Z only (no spaces)."""


@dataclass
class FingerspellDecoder:
    """
    Configurable post-processor for letter streams.

    Parameters
    ----------
    config
        Tunable weights (see ``DecoderConfig``).
    lexicon
        Unigram lexicon; default loads ``data/word_freq_sample.txt``.
    bigram_path
        Optional path to a sparse bigram count file.
    """

    config: DecoderConfig = field(default_factory=DecoderConfig)
    lexicon: WordFrequencyModel | None = None
    bigram_path: Path | None = None

    def __post_init__(self) -> None:
        if self.lexicon is None:
            self.lexicon = load_default_lexicon()
        bp = self.bigram_path
        if bp is None and _DEFAULT_BIGRAMS.is_file():
            bp = _DEFAULT_BIGRAMS
        self._bigrams = load_optional_bigrams(bp)

    def decode_frames(self, frames: Sequence[FramePrediction], *, video_mode: bool = False) -> PipelineResult:
        """Run smoothing → word split → lexical decode → beam search.

        Set ``video_mode=True`` for **sampled** offline clips (e.g. upload); uses a lighter
        smoother that avoids rolling-window majority mixing across letter boundaries.
        """
        sm = (
            smooth_frames_video(frames, self.config)
            if video_mode
            else smooth_frames(frames, self.config)
        )
        letters_for_decode = normalize_fingerspell_letters(sm.letters)
        frags = split_smoothed_into_words(letters_for_decode)
        sentence, beams, lattice = decode_sentence_words(
            frags, self.lexicon, self.config, self._bigrams
        )
        cleaned = "".join(c for c in letters_for_decode.upper() if c.isalpha())
        return PipelineResult(
            raw_smoothed=sm,
            letters_normalized=letters_for_decode,
            word_fragments=frags,
            sentence=sentence,
            beam_hypotheses=beams,
            word_lattice=lattice,
            cleaned_letters_no_spaces=cleaned,
        )

    def decode_noisy_word_string(self, noisy_word: str) -> tuple[str, list[WordCandidate]]:
        """
        Skip temporal smoothing; decode a single noisy letter string (e.g. ``HELXO``).

        Returns (best_word, ranked_candidates).
        """
        cands = decode_word(noisy_word, self.lexicon, self.config)
        if not cands:
            return noisy_word.lower(), []
        return cands[0].word, cands
