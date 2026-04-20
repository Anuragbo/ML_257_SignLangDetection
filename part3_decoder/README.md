# Part 3 — Fingerspelling decoder (letters → words → sentences)

This package is a **post-processor** for any sign-recognition model that outputs per-frame letter hypotheses (top-1 or top-k with probabilities). It does **not** import OpenCV, PyTorch, or `part1_letter_classifier`.

## Pipeline stages

1. **Temporal smoothing** (`temporal_smoothing.py`) — rolling-window vote with a **boost for the current frame**, strict majority when the window is full, **blank frames that do not bleed** old letters, **edge singleton removal** (e.g. `T` before `HHH`), run collapsing, and long blank runs → **spaces** (word boundaries).
2. **Word decoding** (`word_decoder.py`) — score dictionary words with Levenshtein distance, log frequency, visual confidence, length prior, and a small prefix bonus.
3. **Sentence assembly** (`sentence_builder.py` + `beam_search.py`) — combine per-word candidates with beam search; optional sparse **bigrams** (`data/bigrams_sample.txt`).

## Configuration

See `config.py` (`DecoderConfig`): thresholds for smoothing, weights for lexical scoring, beam width, and bigram weight.

## Data files

| File | Purpose |
|------|---------|
| `data/word_freq_sample.txt` | Unigram counts (replace with a larger list for real use) |
| `data/bigrams_sample.txt` | Optional `(word1, word2)` counts for light sentence scoring |

## Commands

From the **repository root**:

```bash
python -m part3_decoder.demo_decode
python -m part3_decoder.demo_decode --word HELXO
```

## Integration (vision model → decoder)

1. For each video frame, build a `FramePrediction` (`letter`, `confidence`, optional `top_k`).
2. Use `frame_from_top1` / `frame_from_top_k` (`frame_utils.py`) if needed.
3. Pass the list to `FingerspellDecoder.decode_frames` (`fingerspell_pipeline.py`).
4. Read `PipelineResult.sentence`, `word_lattice`, and `beam_hypotheses`.

## Scoring (short)

**Word score** (higher is better):

- `weight_visual_confidence * avg_letter_confidence`
- `+ weight_word_frequency * normalized_log_freq(word)`
- `- weight_edit_penalty * levenshtein(noisy, word)`
- `- weight_length_prior * |len(word) - len(noisy)|`
- small prefix bonus when the noisy string matches the start of the candidate

**Beam**: sum of word scores per position; if bigrams are loaded, add `weight_bigram * bigram.score(prev, next)`.

## Tuning tips

- **More smoothing / less flicker**: raise `rolling_window`, `min_majority_votes`, or `pause_blank_frames` (word gaps).
- **Trust the camera more**: raise `weight_visual_confidence`, lower `weight_edit_penalty`.
- **Trust the dictionary more**: raise `weight_word_frequency`, lower `max_edit_distance` (stricter matches).
- **Autocomplete / partial words**: raise `weight_prefix_bonus`; ensure the lexicon contains common stems.
- **Sentence coherence**: add bigram data and raise `weight_bigram` slightly.

## Limits

- Full lexicon scan per word is fine for **thousands** of words; for huge vocabularies, pre-filter candidates (length buckets, prefix trie) before scoring.
- **Double letters** (`HELLO`) and **held letters** (`AAAA` → `A`) are ambiguous from letters alone; use **frame timing** (holds vs. a short gap between two `L` segments) or a higher-level model later.
