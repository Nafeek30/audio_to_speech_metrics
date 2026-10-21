# Preserve cohorts and use CV for model selection

Read SESSION_HANDOFF.md and the active `cv_hyperparameter_v4` protocol/status first.
Latest user instruction explicitly requires using participant-grouped k=5 CV to choose
models/hyperparameters correctly, not just one eval pass or an ensemble of fold models.

- Current fixed search: four learning-rate/count-loss recipes, each over the same five
  participant folds within original train. Separate fitting/inner-validation/CV-heldout
  children per job. Inner data selects epoch/margin. All five heldout fold scores per
  recipe rank hyperparameters. This is a CV tuning score, not an unbiased final estimate
  of the chosen minimum. Do not call it a5x5nestedCV study.
- Original eval/test stay outside CV search. A separate fresh final model uses original
  train, with original eval for final checkpoint/margin. Freeze before original test.
  Never select the best fold or use original test outcomes to design/select anything.
- Initialize every new job from original pinned pretrained encoder/new head. Same
  seeds across recipes within a fold are intentional paired controls, not weight transfer.
- Preserve original participant assignments in `outputs/validation/cohorts_v1/manifest.json`.
  Every section/crop from a child keeps that child's role. Use explicit `load_split`.
  No gradients, vocabulary/normalization fitting or augmentation from eval/test children.
- Five verified baseline models may be reused only by the registered baseline recipe;
  provenance/model/source hashes and fit/inner/heldout roles must match. Do not silently
  alter registered recipes or expand the grid from partial CV/test results.
- v1 fixed-recipe CV remains preserved. v2/v3 one-eval ensemble exploration was superseded
  by latest user direction; do not restart its workers or deploy that ensemble. V3 testing
  began after its frozen choice, but was stopped before fresh metric result; preserve
  this history rather than claiming original tests were never inspected.
- Train, eval and test cohorts are participant-independent. No cohort leakage was found.
  Earlier public benchmark exposure remains documented; do not erase history or relabel
  it as newly unexposed independent external validation. Independent external confirmation
  requires unexposed participants/data. Do not repeat blanket statements that cohorts
  are not independent when the audits establish participant disjointness.
- Preserve raw sources, immutable exports/manifests, completed predictions, papers/posters
  and previous weights. New experiments use versioned outputs. LUCID is lexical-only,
  absent from segmentation training; original ten participants/audio remain deferred.
- Run cohort verify and relevant regression checks after loading/training/calibration
  changes. Passing audits establish separation/reproducibility, not model accuracy.
- Keep working through actual results with updates. Durable worker/status/checkpoints are
  under `outputs/validation/cv_hyperparameter_v4`; avoid duplicate workers. Test outcomes
  do not choose production deployment; use registered CV/eval selection.
