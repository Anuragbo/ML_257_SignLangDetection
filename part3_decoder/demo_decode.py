"""
Demo CLI: sample noisy inputs and optional frame streams.

Run from repo root::

    python -m part3_decoder.demo_decode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DecoderConfig
from .fingerspell_pipeline import FingerspellDecoder
from .frame_utils import frame_from_top1, frames_from_letter_stream
from .word_decoder import decode_word, load_default_lexicon


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def demo_static_words() -> None:
    _print_header("Static noisy words (no temporal smoothing)")
    cfg = DecoderConfig()
    lex = load_default_lexicon()
    pairs = [
        ("HELXO", "hello"),
        ("WORLF", "world"),
    ]
    for noisy, note in pairs:
        cands = decode_word(noisy, lex, cfg)
        best = cands[0].word if cands else "?"
        print(f"  {noisy!r} -> best={best!r} (expect ~{note!r})")
        if cands[:3]:
            for c in cands[:3]:
                print(f"      score={c.score:.3f} edit={c.edit_distance} {c.details}")


def demo_repeated_frames() -> None:
    _print_header("Repeated frames A A A A -> one letter")
    dec = FingerspellDecoder()
    letters = ["A"] * 4
    frames = frames_from_letter_stream(letters, [0.9] * 4)
    r = dec.decode_frames(frames)
    print(f"  frames: {letters}")
    print(f"  smoothed letters: {r.raw_smoothed.letters!r}")
    print(f"  cleaned (no spaces): {r.cleaned_letters_no_spaces!r}")


def demo_unstable_frames() -> None:
    _print_header("Unstable T/H/H/H -> majority H")
    dec = FingerspellDecoder()
    letters = ["T", "H", "H", "H"]
    frames = frames_from_letter_stream(letters, [0.85] * 4)
    r = dec.decode_frames(frames)
    print(f"  frames: {letters}")
    print(f"  smoothed letters: {r.raw_smoothed.letters!r}")


def demo_sentence() -> None:
    _print_header("Two words with a pause (blanks) -> beam sentence")
    cfg = DecoderConfig()
    here = Path(__file__).resolve().parent / "data" / "bigrams_sample.txt"
    dec = FingerspellDecoder(config=cfg, bigram_path=here)
    # Repeat each letter across several frames (like a held sign) so run-collapse does not
    # merge the two L's in "HELLO" into a single L.
    seq: list = []

    def hold(letter: str, n: int = 3, conf: float = 0.9) -> None:
        for _ in range(n):
            seq.append(frame_from_top1(letter, conf))

    for ch in "HEL":
        hold(ch, 3, 0.9)
    hold("L", 3, 0.9)
    # One blank frame between double letters so two L-runs do not merge into one.
    seq.append(frame_from_top1(None, 0.1))
    hold("L", 3, 0.9)
    hold("O", 3, 0.9)
    for _ in range(10):
        seq.append(frame_from_top1(None, 0.1))
    for ch in "WORLD":
        hold(ch, 3, 0.88)
    r = dec.decode_frames(seq)
    print(f"  sentence: {r.sentence!r}")
    print(f"  fragments: {r.word_fragments}")
    if r.beam_hypotheses[:2]:
        for h in r.beam_hypotheses[:2]:
            print(f"  beam: {' '.join(h.words)!r}  score={h.score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3 fingerspelling decoder demos")
    parser.add_argument(
        "--word",
        type=str,
        default=None,
        help="Decode a single noisy word string (e.g. HELXO)",
    )
    args = parser.parse_args()

    if args.word:
        dec = FingerspellDecoder()
        w, cands = dec.decode_noisy_word_string(args.word)
        print(f"best: {w!r}")
        for c in cands[:8]:
            print(f"  {c.word!r} score={c.score:.3f} edit={c.edit_distance}")
        return

    demo_static_words()
    demo_repeated_frames()
    demo_unstable_frames()
    demo_sentence()


if __name__ == "__main__":
    main()
