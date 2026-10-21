# Five-fold participant-grouped hyperparameter selection

Train, eval and test cohorts remain participant-independent. Original train supplies all five-fold hyperparameter comparisons; original eval/test stay outside the search. Each recipe has five separately trained models, with independent fitting/inner-validation/CV-heldout participants within a fold.

| Recipe | Encoder / head LR | Count-loss weight | Pooled CV objective % | Fold mean ± SD % |
|---|---|---:|---:|---:|
| lr2_count1 | 2e-05 / 0.0001 | 1 | 13.16 | 13.16 ± 2.48 |
| lr2_count025 | 2e-05 / 0.0001 | 0.25 | 13.36 | 13.36 ± 2.68 |
| lr3_count1 | 3e-05 / 0.00015 | 1 | 13.52 | 13.53 ± 3.10 |
| lr3_count025 | 3e-05 / 0.00015 | 0.25 | 13.33 | 13.34 ± 2.89 |

Selected **lr2_count1** by the complete five-fold ranking, then trained a separate fresh final model. Original eval selected its epoch and boundary margin before original test scoring.

The CV minimum is a hyperparameter-selection score. Final performance is assessed separately below; selecting on CV does not make the selected CV minimum an unbiased generalization estimate. Fold SD is descriptive, not an independent-fold confidence interval.

| Corpus | Final-test eligible-count APE % | MLUw APE % | Length-SD APE % |
|---|---:|---:|---:|
| nyu_emerson | 23.45 | 14.46 | 13.24 |
| ocsc | 9.81 | 8.75 | 15.02 |

All original cohort assignments, source words, native human references and preceding results are preserved. LUCID lexical results are unchanged and remain separate from boundary-model training. Full native outputs, missingness, predictions, pairwise CSV and participant-bootstrap intervals accompany this report.
