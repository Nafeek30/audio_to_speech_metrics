# Participant-grouped five-fold cross-validation

Five outer folds use only the original training cohort. Each model starts from the original pinned pretrained MiniLM encoder with a newly initialized task head. A separate inner participant holdout selects epochs and the fixed seven-margin calibration; the outer participants never enter either fitting or selection. The original eval and test cohorts retain their assignments and are outside this cross-validation study.

Coverage: 231 participants, 630 sections, one held-out prediction per section. All words are preserved. Results are internal participant-independent validation of the fixed recipe, not evaluation on a newly collected external corpus.

| Corpus | Metric | Pooled participant APE % | Fold mean ± SD % |
|---|---|---:|---:|
| nyu_emerson | eligible_mlu_utterances | 16.40 | 16.40 ± 4.64 |
| nyu_emerson | MLUw | 10.57 | 10.57 ± 2.50 |
| nyu_emerson | length_sd | 20.18 | 20.18 ± 11.89 |
| nyu_emerson | TNW | 3.22 | 3.22 ± 1.02 |
| nyu_emerson | NDW | 2.72 | 2.72 ± 1.11 |
| nyu_emerson | TTR | 1.64 | 1.64 ± 0.63 |
| nyu_emerson | MATTR | 1.57 | 1.61 ± 0.77 |
| ocsc | eligible_mlu_utterances | 9.16 | 9.17 ± 0.74 |
| ocsc | MLUw | 9.09 | 9.10 ± 1.09 |
| ocsc | length_sd | 14.89 | 14.90 ± 3.15 |
| ocsc | TNW | 1.51 | 1.51 ± 0.18 |
| ocsc | NDW | 1.67 | 1.68 ± 0.13 |
| ocsc | TTR | 1.00 | 1.00 ± 0.15 |
| ocsc | MATTR | 0.59 | 0.59 ± 0.09 |

Fold SD describes variation across folds; it is not a confidence interval from independent fold estimates. Zero-reference denominators and unavailable SD/MATTR remain missing in detailed summaries. Native CLAN reference and predicted outputs are retained.

| Corpus | Boundary precision % | Recall % | F1 % | Exact span F1 % |
|---|---:|---:|---:|---:|
| nyu_emerson | 84.54 | 93.54 | 88.81 | 71.57 |
| ocsc | 87.84 | 86.60 | 87.22 | 68.41 |

## Cohort and history checks

No original train/eval/test assignment was altered. Original model/results are archived under previous_production. The preceding run froze its model at 2026-10-15T16:18:40.030077+00:00 before its recorded final evaluation at 2026-10-15T16:18:43.999852+00:00. Those cohorts were participant-disjoint; previous protocols also disclose earlier benchmark inspection. The new CV adds held-out predictions from fresh task training without erasing that history.
