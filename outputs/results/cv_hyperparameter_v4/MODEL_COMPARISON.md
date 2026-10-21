# Frozen final model versus previous production

Descriptive paired comparison on the preserved historically evaluated test cohort. Reporting only; no selection or deployment feedback. Bootstrap resamples children, conditional on these fitted models; it does not include training or tuning uncertainty.

| Corpus | Measure | Previous APE % | Final APE % | Paired change pp [95% bootstrap interval] | Missing sections, previous/final |
|---|---|---:|---:|---:|---:|
| nyu_emerson | eligible_mlu_utterances | 29.98 | 23.45 | -6.53 [-11.22, -1.85] | 0/0 |
| nyu_emerson | MLUw | 16.57 | 14.46 | -2.11 [-4.88, +1.20] | 0/0 |
| nyu_emerson | length_sd | 13.70 | 13.24 | -0.46 [-4.88, +2.79] | 0/0 |
| ocsc | eligible_mlu_utterances | 10.36 | 9.81 | -0.56 [-1.40, +0.30] | 0/0 |
| ocsc | MLUw | 8.70 | 8.75 | +0.05 [-0.98, +1.06] | 0/0 |
| ocsc | length_sd | 14.24 | 15.02 | +0.78 [-0.99, +2.24] | 2/2 |

Negative changes indicate lower error. Paired changes use sections available for both methods; individual APE columns use each method’s available sections. NYU has only four test children, so its uncertainty must be interpreted cautiously.

All 126 sections have identical words and human reference values. Exact lexical parity by measure: TNW: True, NDW: True, TTR: True, MATTR: True.

The CV ranking and final test have different roles and training sample sizes. A lower CV selection score alone is not evidence of better production accuracy.
