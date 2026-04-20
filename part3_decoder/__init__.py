"""
Part 3 — Fingerspelling **decoder** (letters → words → sentences).

Independent of any particular vision model; feed ``FramePrediction`` streams from CNN/YOLO/etc.
"""

from .beam_search import BeamHypothesis, beam_search_word_lists
from .config import DecoderConfig, DecoderRuntime
from .fingerspell_pipeline import FingerspellDecoder, PipelineResult
from .frame_utils import frame_from_top1, frame_from_top_k, frames_from_letter_stream
from .language_utils import (
    BigramModel,
    WordFrequencyModel,
    levenshtein,
    normalize_fingerspell_letters,
    normalize_letters,
)
from .sentence_builder import decode_sentence_words, load_optional_bigrams, split_smoothed_into_words
from .temporal_smoothing import FramePrediction, SmoothedOutput, smooth_frames, smooth_frames_video
from .word_decoder import WordCandidate, decode_word, load_default_lexicon

__all__ = [
    "BeamHypothesis",
    "BigramModel",
    "DecoderConfig",
    "DecoderRuntime",
    "FingerspellDecoder",
    "FramePrediction",
    "PipelineResult",
    "SmoothedOutput",
    "WordCandidate",
    "WordFrequencyModel",
    "beam_search_word_lists",
    "decode_sentence_words",
    "decode_word",
    "frame_from_top1",
    "frame_from_top_k",
    "frames_from_letter_stream",
    "levenshtein",
    "load_default_lexicon",
    "load_optional_bigrams",
    "normalize_fingerspell_letters",
    "normalize_letters",
    "smooth_frames",
    "smooth_frames_video",
    "split_smoothed_into_words",
]
