# Public common-input benchmark

All four methods receive the same unpunctuated target-child paragraphs, without source-turn cuts. References are frozen human CHAT units on corrected spoken words. No original ten-participant data were accessed.

Primary: mean APE across utterance count, surface MLUw and population length SD, recording-balanced within child, child-balanced within corpus and equally weighted across corpora. Lower is better.

| Method | Corpus | Children | Paragraphs | Count APE % | MLUw APE % | SD APE % | Primary % | Boundary F1 % | Exact span F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | nyu_emerson | 4 | 15 | 66.44 | 166.36 | 257.35 | 163.39 | 38.78 | 12.61 |
| rules | ocsc | 45 | 111 | 64.61 | 182.71 | 267.25 | 171.52 | 43.12 | 13.80 |
| previous_minilm | nyu_emerson | 4 | 15 | 32.33 | 19.66 | 21.11 | 24.37 | 79.39 | 54.34 |
| previous_minilm | ocsc | 45 | 111 | 10.58 | 8.35 | 15.43 | 11.46 | 86.64 | 67.32 |
| latest_minilm | nyu_emerson | 4 | 15 | 24.91 | 16.99 | 18.87 | 20.26 | 76.92 | 52.18 |
| latest_minilm | ocsc | 45 | 111 | 10.79 | 8.75 | 16.20 | 11.91 | 85.56 | 65.14 |
| batchalign_0.10.4 | nyu_emerson | 4 | 15 | 56.48 | 106.04 | 77.40 | 79.97 | 49.20 | 14.26 |
| batchalign_0.10.4 | ocsc | 45 | 111 | 52.47 | 113.64 | 121.48 | 95.86 | 56.31 | 23.03 |

## Corpus-balanced primary comparison

| Method | Mean APE % | Difference vs Batchalign (95% paired child interval) |
|---|---:|---|
| rules | 167.45 | +79.53 [+67.41, +92.11] |
| previous_minilm | 17.91 | -70.01 [-77.24, -62.88] |
| latest_minilm | 16.09 | -71.83 [-79.43, -64.63] |
| batchalign_0.10.4 | 87.92 | Reference competitor |

## Lexical metric errors and coverage

| Method | Corpus | TNW APE % | NDW APE % | TTR APE % | MATTR50 APE % | Missing MATTR50 |
|---|---|---:|---:|---:|---:|---:|
| rules | nyu_emerson | 33.05 | 26.70 | 12.19 | 4.58 | 4 |
| rules | ocsc | 11.75 | 9.90 | 2.65 | 0.89 | 8 |
| previous_minilm | nyu_emerson | 7.79 | 7.97 | 2.74 | 2.69 | 0 |
| previous_minilm | ocsc | 0.83 | 0.64 | 0.44 | 0.33 | 7 |
| latest_minilm | nyu_emerson | 6.69 | 6.90 | 2.60 | 2.87 | 0 |
| latest_minilm | ocsc | 0.81 | 0.63 | 0.43 | 0.34 | 7 |
| batchalign_0.10.4 | nyu_emerson | 22.06 | 14.90 | 10.39 | 4.59 | 3 |
| batchalign_0.10.4 | ocsc | 5.28 | 4.08 | 1.73 | 0.75 | 7 |

Word streams preserved in 504/504 method-recording pairs. Entire eligible gap masks and raw official inputs/outputs are retained. Exact spans use all original word positions; boundary F1 masks final and unintelligible-adjacent gaps.

Native CLAN calculator checks: {'expected': 30, 'computed': 30, 'passed': True}. These check normalized arithmetic, not agreement of predicted boundaries with professionals. Raw outputs are in the validation folder.

## Interpretation limits

Batchalign 0.10.4 official CHATUtteranceBackend and its pinned English model are evaluated as a segmentation component. This is not a rerun of the complete 2023 published audio pipeline. Existing tests have historical exposure. Our MiniLM models were trained on disjoint children from these corpus domains; the official pretrained component was not adapted here. This is not an architecture-only or equal-adaptation comparison. NYU has only four test children. These are descriptive matched-data results, with no model selection or tuning. Normalized spoken-word references differ from full original CHAT coding; retraces, morphology and professional review remain separate. Lexical errors can change through utterance-level unintelligibility exclusions even with unchanged words. MLUm, WPM, PGU, production errors and mazes remain unavailable without validated corresponding predictions. MOSS audio transcription, target-speaker identification and audio-to-metric accuracy have not been validated by this experiment.

Files: `results.json` contains per-recording signed/absolute percentage errors, predictions, hashes, coverage and uncertainty; `batchalign_raw.jsonl` contains official UtSeg inputs/outputs. Protocol and CLAN evidence: `outputs/validation/common_benchmark/`.
