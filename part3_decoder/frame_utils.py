"""
Helpers to build ``FramePrediction`` lists from model outputs (top-1 or top-k).
"""

from __future__ import annotations

from typing import Sequence

from .temporal_smoothing import FramePrediction


def frame_from_top1(letter: str | None, confidence: float = 1.0) -> FramePrediction:
    """Single timestep from the vision model's argmax letter."""
    let = letter.upper().strip() if letter else None
    if let is not None and len(let) != 1:
        let = let[0] if let else None
    return FramePrediction(letter=let, confidence=float(confidence), top_k=[])


def frame_from_top_k(top_k: list[tuple[str, float]], blank_token: str | None = None) -> FramePrediction:
    """
    One timestep from a distribution over letters.

    ``top_k`` is sorted by probability descending; the first entry is used as top-1.
    ``blank_token`` (e.g. ``\"_\"`` or ``\"blank\"``) is mapped to ``letter=None``.
    """
    if not top_k:
        return FramePrediction(letter=None, confidence=0.0, top_k=[])
    let, conf = top_k[0]
    let = let.strip()
    if blank_token is not None and let.upper() == blank_token.upper():
        return FramePrediction(letter=None, confidence=float(conf), top_k=top_k)
    if not let:
        return FramePrediction(letter=None, confidence=float(conf), top_k=top_k)
    ch = let[0].upper()
    return FramePrediction(letter=ch, confidence=float(conf), top_k=top_k)


def frames_from_letter_stream(
    letters: Sequence[str | None],
    confidences: Sequence[float] | None = None,
) -> list[FramePrediction]:
    """
    Build a frame list from parallel arrays (e.g. after reading a CSV of predictions).

    If ``confidences`` is omitted, each frame uses confidence ``1.0``.
    """
    out: list[FramePrediction] = []
    for i, L in enumerate(letters):
        conf = 1.0 if confidences is None else float(confidences[i])
        out.append(frame_from_top1(L, conf))
    return out
