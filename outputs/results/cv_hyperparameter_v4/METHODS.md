# Study status: stopped at the user’s request

Three configurations completed all five folds. The fourth stopped during its first fold. We exclude that configuration entirely from ranking. The baseline has the best completed combined CV score; lower count-loss weight gives the best completed OCSC mean. Existing production remains active. No new v4 final fit, final test or activation occurred. The registered protocol below describes the full planned procedure; its final stages did not run.

# Participant-grouped CV for hyperparameter selection

The aim is to improve the production training procedure using cross-validated model
selection. Cross-validation does not change the loss by itself: the candidate learning
rates and count-loss weights change training, and CV compares those resulting models.

## Preserved cohorts

| Source | Original train: participants / sections | Original eval | Original test |
|---|---:|---:|---:|
| OCSC | 211 / 567 | 45 / 125 | 45 / 111 |
| NYU–Emerson | 20 / 63 | 4 / 14 | 4 / 15 |

Counts refer to usable data. Original assignments, the four documented source
exclusions, all 333 CHAT files, and all 895 prepared sections remain unchanged.
Every section and crop from a child keeps the child's role. LUCID is a separate
lexical benchmark, not segmentation-training data. Audio and the original ten
participants remain deferred.

## Fixed 2-by-2 training grid

| Configuration | Encoder LR | Head/embedding LR | Count-loss coefficient |
|---|---:|---:|---:|
| lr2_count1 | 0.00002 | 0.00010 | 1 |
| lr2_count025 | 0.00002 | 0.00010 | 0.25 |
| lr3_count1 | 0.00003 | 0.00015 | 1 |
| lr3_count025 | 0.00003 | 0.00015 | 0.25 |

All other architecture, feature, augmentation and optimization settings are fixed.
The encoder is original pinned Microsoft MiniLM-L12-H384-uncased. Its contextual
left/right gap representations plus a learned 16-dimensional rule embedding enter
a 128-unit GELU/dropout classifier. Encoder, rule embedding and classifier are
fine-tuned jointly. No previous fine-tuned task weights initialize a new job.

Training uses cross entropy plus the candidate coefficient times relative boundary
count loss. Fixed settings: dropout0.15, AdamW weight decay0.01, batch8/accumulation2,
10% warmup, gradient clipping1, six epochs maximum, patience2, overlapping token
windows256/stride128, one seeded word-position crop per fit paragraph, final crop
gap masked, 5% non-special UNK corruption, equal expected corpus sampling mass.
All crop parents and labels are checked against fitting participants only.

## Five participant-grouped folds per recipe

The same original-training participant folds are used for all four configurations.
Each outer CV fold holds out46–47 children, with all their sections. The remaining
original-training children are separated into fitting147–148 and inner-validation37.
Fitting children supply gradients; inner children select checkpoint and margin;
CV-heldout children supply the configuration-ranking scores.

Each configuration therefore has five independently fitted fold models. Shared seeds
within a fold across configurations intentionally pair initialization, crops and noise
for a controlled comparison. This is fresh initialization from the original encoder,
not continuation from another configuration's trained weights. Outer-heldout children
are excluded from every fitting/inner selection operation for that fold.

Five already completed baseline fits can be reused because their fitting recipe,
initialization, participant roles and weights are verified. Their inner calibration is
rerun with the same grid used for every v4 configuration. No old outer score selects
a margin. Fifteen new configuration/fold models must actually be trained. Previous
baseline predictions/results remain preserved under their original protocol.

## Checkpoint, margin and recipe selection

Within each fold, select minimum inner native CLAN objective at margin0, with earlier
epochs winning exact ties. Reload that checkpoint and verify prediction parity. Then
select the minimum inner objective over the fixed margins in listed tie order:
0, -0.125, +0.125, -0.25, +0.25, -0.5, +0.5, -1, +0.75, +1, +1.5, +2.
The order prefers no/small changes on ties. Freeze before CV-heldout scoring.

Each recipe must finish all five folds before ranking. Pool exactly one heldout
prediction per original-training section per recipe. Compute APE within section,
mean within participant, mean within corpus, then equally average eligible utterance
count, MLUw and length SD across the two corpora. Minimum pooled CV objective chooses
the recipe; exact ties keep registered recipe order. Report all five fold values,
fold mean and sample SD, corpus/metric results and missingness. No best-fold deployment,
partial-result grid expansion or external-test tuning is allowed.

These CV scores are used to tune hyperparameters. The selected minimum is a tuning
score, not an unbiased final performance estimate of the selected procedure. This
is a five-fold hyperparameter search with a within-fold inner holdout; it is not
claimed to be a5x5nested-CV experiment. Fold SD is descriptive, since training sets
overlap. The external original test supplies the separate final assessment.

## Fresh final fit and runtime

After the full CV ranking is frozen, initialize another fresh model using the chosen
recipe. Gradients use original train only; original eval selects checkpoint/margin
under the same fixed policy. Freeze final weight hash and margin, then score original
test. Original eval/test do not select the recipe from the four training settings.
Production follows this registered CV/eval procedure, never test-score comparisons.
All139 original-eval runtime predictions and actual CLI/native-metric examples from
both corpora must match before activation; preserve/restore prior workflow on failure.

Native CLAN MLU/FREQ calculations, source words and human reference values are fixed.
Predictions do not fabricate retrace, morphology, grammaticality or maze annotations.
Zero denominators/unavailable values remain missing. Word preservation, native raw
outputs, hashes, cohort roles, initialization, crops, timing and summaries are audited.

## History and scope

The preceding five-fold fixed-recipe baseline remains valid as that fixed experiment.
The subsequent one-eval ensemble search was superseded when the user clarified the
need for CV-driven hyperparameter selection. That branch froze a choice, began test
preparation/emission caching, and was stopped before saving a fresh metric result.
Its partial artifacts and earlier benchmark exposure remain documented; they do not
enter this hyperparameter ranking. Train/eval/test remain participant-independent.
This study does not claim a newly collected external corpus or erased exposure history.

Primary methodological references: [scikit-learn hyperparameter search](https://scikit-learn.org/stable/modules/grid_search.html)
and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
