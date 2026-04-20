"""
Temporal smoothing for frame-level letter streams from any vision backend.

Input: sequence of per-frame predictions (letter + confidence, optional top-k).
Output: stabilized letter sequence with optional word-boundary markers (spaces).

This module does not import OpenCV, torch, or project Part 1 code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .config import DecoderConfig


def _demote_edge_singleton_letter_runs(robust: list[str | None]) -> list[str | None]:
    """
    Remove a **single** leading or trailing letter when it disagrees with the next (or prev)
    longer run — e.g. ``T H H H`` → ``H H H`` so collapse yields ``H`` instead of ``TH``.

    This is a lightweight "spike" filter for unstable first/last frames.
    """
    r = list(robust)

    def first_letter_run() -> tuple[int, int, str] | None:
        i = 0
        while i < len(r) and r[i] is None:
            i += 1
        if i >= len(r):
            return None
        j = i
        ch = r[i]
        while j < len(r) and r[j] == ch:
            j += 1
        return (i, j, ch)

    def next_letter_run(start: int) -> tuple[int, int, str] | None:
        i = start
        while i < len(r) and r[i] is None:
            i += 1
        if i >= len(r):
            return None
        j = i
        ch = r[i]
        while j < len(r) and r[j] == ch:
            j += 1
        return (i, j, ch)

    a = first_letter_run()
    if a is not None:
        i0, i1, ch_a = a
        if i1 - i0 == 1:
            b = next_letter_run(i1)
            if b is not None:
                j0, j1, ch_b = b
                if ch_a != ch_b and (j1 - j0) >= 2:
                    r[i0] = None

    # Trailing singleton (mirror)
    runs: list[tuple[int, int, str]] = []
    idx = 0
    while idx < len(r):
        while idx < len(r) and r[idx] is None:
            idx += 1
        if idx >= len(r):
            break
        j = idx
        ch = r[idx]
        while j < len(r) and r[j] == ch:
            j += 1
        runs.append((idx, j, ch))
        idx = j
    if len(runs) >= 2:
        p0, p1, ch_p = runs[-2]
        l0, l1, ch_l = runs[-1]
        if (l1 - l0) == 1 and (p1 - p0) >= 2 and ch_p != ch_l:
            for k in range(l0, l1):
                r[k] = None

    return r


@dataclass
class FramePrediction:
    """One time step from the sign-recognition model (agnostic of which model)."""

    letter: str | None
    """None = blank / no reliable letter."""

    confidence: float = 1.0
    top_k: list[tuple[str, float]] = field(default_factory=list)

    def effective_letter(self, min_confidence: float) -> str | None:
        if self.letter is None or self.confidence < min_confidence:
            return None
        return self.letter.upper()


@dataclass
class SmoothedOutput:
    """Result of temporal processing."""

    letters: str
    """Letters and spaces (spaces = word boundaries from long pauses)."""

    letter_confidences: list[float]
    """Per emitted character (letter or space placeholder 0.0)."""

    raw_collapsed: str
    """Same letters without pause spaces (debug / alternate display)."""


def _emit_from_robust(
    robust: list[str | None],
    frames: Sequence[FramePrediction],
    config: DecoderConfig,
) -> SmoothedOutput:
    """Turn a per-frame letter sequence into collapsed letters + pause spaces."""
    out: list[str] = []
    confs: list[float] = []
    raw_letters: list[str] = []

    i = 0
    blank_run = 0
    while i < len(robust):
        ch = robust[i]
        if ch is None:
            blank_run += 1
            i += 1
            if blank_run >= config.pause_blank_frames:
                if out and out[-1] != " ":
                    out.append(" ")
                    confs.append(0.0)
                blank_run = 0
            continue

        blank_run = 0
        j = i
        while j < len(robust) and robust[j] == ch:
            j += 1
        run_len = j - i
        if run_len < config.min_consistent_frames:
            i = j
            continue
        sub_frames = frames[i:j]
        avg_c = sum(f.confidence for f in sub_frames) / len(sub_frames)
        out.append(ch)
        confs.append(avg_c)
        raw_letters.append(ch)
        i = j

    letters_str = "".join(out)
    raw_str = "".join(raw_letters)

    return SmoothedOutput(letters=letters_str, letter_confidences=confs, raw_collapsed=raw_str)


def smooth_frames(frames: Sequence[FramePrediction], config: DecoderConfig) -> SmoothedOutput:
    """
    Stabilize a frame stream:

    1. Map each frame to an effective letter (or None if low confidence).
    2. **Blank frames** stay blank (older letters do not leak through the window).
    3. Replace each index with the **majority vote** over the last ``rolling_window`` non-blank
       frames, with **+2 votes for the current frame** so new letters are not buried by the tail
       of the previous letter, and tie-breaks favor the live frame.
    4. Optionally require ``min_majority_votes`` when the window is full.
    5. **Edge singleton removal** (e.g. one ``T`` before ``HHH``) to reduce unstable first frames.
    6. Each **maximal run** of the same letter becomes **one** emission (A A A A -> A).
    7. Runs shorter than ``min_consistent_frames`` are skipped (noise spikes).
    8. Long runs of None (length >= ``pause_blank_frames``) insert a **space** (word boundary).

    Double letters (e.g. ``HELLO``) need either multiple frames per letter with a **short gap**
    between the two L segments or a richer model; see ``part3_decoder/README.md``.
    """
    if not frames:
        return SmoothedOutput("", [], "")

    eff: list[str | None] = [f.effective_letter(config.min_confidence) for f in frames]
    W = max(1, config.rolling_window)
    robust: list[str | None] = []
    for i in range(len(eff)):
        # A blank/low-confidence frame must stay blank; do not let older letters "bleed"
        # through the rolling window (otherwise pauses never reach ``pause_blank_frames``).
        if eff[i] is None:
            robust.append(None)
            continue
        lo = max(0, i - W + 1)
        chunk = [x for x in eff[lo : i + 1] if x is not None]
        if not chunk:
            robust.append(None)
        else:
            ctr = Counter(chunk)
            # Favor the **current** frame so a new letter (e.g. D after LLL) is not outvoted
            # by the tail of the previous letter in the rolling window.
            cur = eff[i]  # not None: blank frames handled above
            ctr[cur] += 2
            letter, votes = ctr.most_common(1)[0]
            # Break ties (e.g. [T,H] at 50/50) using the **most recent** frame in the window,
            # so ``T H H H`` leans toward ``H`` instead of lengthening ``T``.
            top = ctr.most_common(2)
            if len(top) == 2 and top[0][1] == top[1][1]:
                letter = cur
                votes = ctr[letter]
            need = config.min_majority_votes
            # Strict threshold only when the vote is taken over a full window (stable tail).
            # At the start of the clip, ``chunk`` may be shorter — use a plain majority.
            if len(chunk) >= W and votes < need:
                robust.append(None)
            else:
                robust.append(letter)

    robust = _demote_edge_singleton_letter_runs(robust)

    return _emit_from_robust(robust, frames, config)


def smooth_frames_video(frames: Sequence[FramePrediction], config: DecoderConfig) -> SmoothedOutput:
    """
    Smoothing for **uniformly sampled** video (upload / offline clips).

    Skips rolling-window majority voting and edge singleton demotion. Those steps assume a
    dense live stream where the same letter spans many frames; on sampled video they smear
    boundaries (e.g. ``LD`` → ``L`` or merged runs) and can drop valid edge letters.

    Still applies confidence gating, run collapse, ``min_consistent_frames``, and pause blanks.
    """
    if not frames:
        return SmoothedOutput("", [], "")
    eff: list[str | None] = [f.effective_letter(config.min_confidence) for f in frames]
    return _emit_from_robust(list(eff), frames, config)


def collapse_repeated_letters(s: str) -> str:
    """Collapse runs inside each whitespace-separated word: AAAABBB -> AB."""
    if not s:
        return ""
    parts = s.split()
    out_words: list[str] = []
    for w in parts:
        if not w:
            continue
        cur = [w[0]]
        for c in w[1:]:
            if c != cur[-1]:
                cur.append(c)
        out_words.append("".join(cur))
    return " ".join(out_words)


def letters_only_no_spaces(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalpha())
