# New pipelines versus existing ten-participant results

The established pipeline is substantially closer to the manual participant values than either paragraph model. The new paragraph model oversegments all ten participants, reducing MLUw and length SD. The earlier modest improvement in the mean recording objective does not imply improvement in these pooled-participant median errors.

All three methods use the same ASR source recordings. Established rules retain observed speaker turns; paragraph models withhold those cuts. Models were frozen before this comparison; no training or selection used these results.

The manual comparison pools the same 14 labeled recordings within each of the 10 participants, matching the existing complete-sample analysis. The corrected CLAN/TAALED comparison uses all 37 recordings and the original saved reference values; it includes transcription and reference-boundary differences.

Errors below are median participant APE, in percent; lower is better. For corrected references, first average recording APE within each participant.

| Reference / metric | Established rules | Previous paragraph | New paragraph |
| --- | ---: | ---: | ---: |
| Manual complete participant samples: total_utterances | 6.31 | 38.07 | 39.00 |
| Manual complete participant samples: mlu_words | 6.40 | 27.39 | 28.24 |
| Manual complete participant samples: mlu_words_sd | 16.76 | 22.38 | 23.64 |
| Manual complete participant samples: normalized/basic_ntokens | 0.00 | 0.00 | 0.00 |
| Manual complete participant samples: normalized/basic_ntypes | 0.30 | 0.85 | 0.79 |
| Manual complete participant samples: normalized/simple_ttr_aw | 0.30 | 0.85 | 0.79 |
| Manual complete participant samples: normalized/mattr50_aw | 0.08 | 0.19 | 0.12 |
| Corrected CLAN: mlu_nutterances | 3.38 | 50.43 | 45.00 |
| Corrected CLAN: mlu_words | 3.77 | 30.81 | 29.06 |
| Corrected CLAN: mlu_words_sd | 4.73 | 42.59 | 39.17 |
| Corrected TAALED: basic_ntokens | 3.71 | 3.71 | 3.71 |
| Corrected TAALED: basic_ntypes | 2.39 | 2.78 | 2.58 |
| Corrected TAALED: simple_ttr_aw | 2.26 | 2.16 | 2.48 |
| Corrected TAALED: mattr50_aw | 0.96 | 0.94 | 0.89 |

## Individual participants: human → new pipeline (APE)

| Participant | Utterance count | MLUw | Length SD |
| --- | ---: | ---: | ---: |
| P1 | 111.00 → 161.00 (45.05%) | 11.27 → 7.77 (31.06%) | 6.63 → 5.56 (16.08%) |
| P2 | 136.00 → 220.00 (61.76%) | 8.72 → 5.58 (36.02%) | 5.72 → 4.05 (29.16%) |
| P3 | 107.00 → 158.00 (47.66%) | 10.01 → 6.72 (32.91%) | 8.05 → 4.55 (43.45%) |
| P4 | 113.00 → 138.00 (22.12%) | 4.26 → 3.34 (21.55%) | 3.87 → 2.78 (28.09%) |
| P5 | 119.00 → 164.00 (37.82%) | 5.60 → 4.06 (27.44%) | 4.65 → 3.50 (24.86%) |
| P6 | 122.00 → 150.00 (22.95%) | 5.23 → 4.35 (16.78%) | 3.79 → 3.15 (16.72%) |
| P7 | 121.00 → 126.00 (4.13%) | 2.97 → 2.74 (8.00%) | 2.17 → 1.89 (13.07%) |
| P8 | 112.00 → 157.00 (40.18%) | 7.95 → 5.64 (29.03%) | 6.06 → 5.13 (15.30%) |
| P9 | 95.00 → 170.00 (78.95%) | 13.74 → 7.60 (44.71%) | 9.62 → 4.77 (50.41%) |
| P10 | 111.00 → 126.00 (13.51%) | 5.17 → 4.56 (11.90%) | 4.53 → 3.52 (22.43%) |

Each participant’s reference, predicted value, signed direction (recoverable from predicted minus reference), absolute error and APE are in `metric_pairs.csv`; all measures and both mean/median summaries are in `metric_summary.csv`. Exact per-record boundary counts are in `boundary_recordings.csv`.

Source-word preservation and exact agreement with the previously frozen new-model predictions were checked for all 14 labeled recordings. Native CLAN agreement verifies arithmetic on predicted text, not agreement with human boundaries. Missing morphology, timing, grammaticality, error and maze annotations remain unavailable. Historical benchmark exposure prevents a fresh confirmatory claim.

The full new-model run completed for 37 ASR and 37 corrected recordings, covering all 10 participants in each condition. Runtime auditing verified all 74 native CLAN checks within displayed precision, source-word preservation, 17 TAALED comparisons per recording, and exact reproduction of all 14 frozen labeled-recording predictions. TAALED preprocessing differences are retained rather than described as exact calculator agreement. Publication files were unchanged.
