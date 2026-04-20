"""
Decoder configuration: tunable weights and thresholds for the fingerspelling pipeline.

All vision-model outputs are consumed only as structured inputs; this module has no
dependency on CNN/YOLO/MediaPipe code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecoderConfig:
    """Weights and limits for temporal smoothing, word decoding, beam search, and sentences."""

    # --- Temporal smoothing ---
    min_confidence: float = 0.25
    """Ignore frame predictions below this confidence (treated like blank)."""

    min_consistent_frames: int = 1
    """
    After majority voting, ignore **short** runs of the same letter (noise spikes).
    Runs shorter than this are dropped (set to 1 if each letter is only 1 frame).
    Raise to 2–3 when the vision model emits many duplicate frames per letter.
    """

    rolling_window: int = 5
    """Window size for majority vote when resolving unstable frames (e.g. T/H/H/H)."""

    min_majority_votes: int = 3
    """Minimum votes within the rolling window needed to output a letter."""

    collapse_run_length: bool = True
    """Collapse immediate repeats: A A A -> A (after frame-level stabilization)."""

    pause_blank_frames: int = 8
    """Treat this many consecutive blank/low-confidence frames as a word boundary (space)."""

    # --- Word decoding ---
    max_edit_distance: int = 2
    """Max Levenshtein distance when matching noisy strings to dictionary words."""

    max_word_candidates: int = 50
    """Cap dictionary neighbors considered per noisy word."""

    weight_visual_confidence: float = 1.0
    """Scale for average letter confidence (when provided)."""

    weight_word_frequency: float = 0.35
    """Weight for log(1 + word_frequency)."""

    weight_edit_penalty: float = 1.2
    """Penalty per edit operation vs. noisy string."""

    weight_prefix_bonus: float = 0.15
    """Small bonus when a candidate is a prefix of a high-frequency word (autocomplete feel)."""

    weight_length_prior: float = 0.02
    """Prefer candidates with length close to noisy string length (small)."""

    # --- Beam search ---
    beam_width: int = 8
    """Number of hypotheses to keep when combining word candidates."""

    # --- Sentence assembly ---
    weight_bigram: float = 0.25
    """If bigrams are loaded, bonus for likely (prev_word, next_word) pairs."""

    sentence_context_window: int = 1
    """How many previous words influence scoring (1 = bigram only)."""


@dataclass
class DecoderRuntime:
    """Mutable runtime state (e.g. bigram cache); optional hook for future LM."""

    extra: dict = field(default_factory=dict)
