# Participant cohort audit

No participant, source recording, source-file hash or exact full task-section text crosses the saved train/validation/test cohorts. The original assignments were preserved. No retraining is needed to repair detected cohort overlap because none was found.

All 895 prepared records and their ordering are reproduced exactly by the isolated exports. All 333 raw CHAT source hashes, the selected model weights and the published LUCID table remain unchanged. The 139 validation reference values match the original native CLAN references exactly.

| Corpus | Train children / sections | Eval children / sections | Test children / sections |
|---|---:|---:|---:|
| ocsc | 211 / 567 | 45 / 125 | 45 / 111 |
| nyu_emerson | 20 / 63 | 4 / 14 | 4 / 15 |

Eval means validation/model selection. The original manifest assigns 22 NYU children to train, but two have no eligible prepared speech; 20 contribute records. Two OCSC sources contain no selected discourse task and are outside these exports. All four reasons remain in the manifest.

## Evidence checked

- Participant assignments agree across the original split, improvement protocol, reassessment and metric-training plans.
- The two retained augmentation caches contain 1,260 and 1,923 records respectively; every original/crop belongs to training, and each crop matches its training parent words and labels.
- The calibration pre-selection cache and every candidate-selection result contain validation participants only.
- Isolated loaders validate manifest/file hashes, cohort membership, source ownership, duplicate records and explicit role requests. Calibration selection loads validation references only.
- Full regression results are in regression_tests.log. Tests deliberately inject forbidden participants, foreign crops, altered labels/files/manifests and a cross-role path; all are rejected.

## Interpretation and future use

Train is for fitting; eval is for selecting models, epochs, workflows and margins; test is for evaluating a frozen choice. Test results must not select future changes. Original test participants must never be moved into train/eval.

The existing test cohort was historically exposed. It remains useful for reproducible descriptive evaluation; separating or retraining on the same split cannot make it an untouched independent test. Independent confirmation requires genuinely unexposed participants/data.

LUCID stays outside these ML cohorts as a separate, historically exposed lexical-comparison benchmark. The original ten participants remain outside this export. No new data, training, model selection or test scoring was performed in this audit.
