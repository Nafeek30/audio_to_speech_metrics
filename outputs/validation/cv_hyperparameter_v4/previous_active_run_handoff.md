# Current task: CV-driven model/hyperparameter improvement

User explicitly clarified that k-fold CV should support reliable model/hyperparameter
selection, not be treated as an accuracy-improving operation itself. The earlier
fixed-recipe five-fold baseline was proper as fixed-recipe validation, but one-eval
ensemble search did not answer this goal. That branch was stopped; no fresh final
metric result was saved and no production replacement occurred. Original train,
eval and test cohorts are participant-independent and unchanged.

## Active v4 search — continue to results, not just status

Module `speech_pipeline.research.cv_hyperparameter`; immutable protocol
`outputs/validation/cv_hyperparameter_v4/protocol.json`. Four fixed recipes:
encoder LR2e-5/head1e-4 or encoder3e-5/head1.5e-4, crossed with count-loss weight .25 or1.
Same participant-grouped5folds inside original TRAIN, same paired seeds/initial
heads per fold across recipes. Each job independently initializes original pinned
MiniLM encoder/new head; paired same seed does NOT mean transferred task weights.
Each fold's fit/inner/heldout children stay separate. Inner selects epoch (native
objective at margin0; max6/patience2), then twelve fixed margins (ties favor0).
Only complete five-fold pooled heldout scores choose recipe. Every recipe must finish
all5folds; no best-fold deployment or partial-result grid expansion. CV scores are
hyperparameter-selection scores, not unbiased final estimates of the selected minimum.

Original eval/test stay OUTSIDE CV recipe selection. After frozen recipe choice,
fresh final model fits original train, original eval selects epoch/margin under the
same policy, then freeze before original-test scoring. Production choice follows
CV-selected recipe and eval-selected checkpoint/margin, never test scores. Earlier
v2/v3 one-eval deployment guard is superseded; no ensemble deployment in this study.

The five completed baseline fold fits are reused with hashes/roles checked and
inner calibration rerun identically to other recipes. Fifteen new fold fits plus a
fresh final fit remain. Training hyperparameters are actually varied and validated,
not just repeatedly scoring the same weights. Gradients/crops use only fit children.

Live status/PIDs: `outputs/validation/cv_hyperparameter_v4/job_status.json`.
Main supervisor launched PID15876; inspect live file rather than trusting snapshot.
That supervisor was intentionally stopped after lower-count fold2 epoch6 was saved.
Native CLAN startup was forking an18GB training process repeatedly. A launch-only
optimization in corpora.py now permits macOS posix_spawn (absolute input paths,
same binary/arguments/PTY/parser). All139 eval reference native values and119 tests
match. Details/hashes/checkpoint in clan_launch_amendment.json. Resumed supervisor
PID78544; protected model/protocol sources unchanged. Inspect live status for PIDs.
Logs: `cv_training_selection_and_test.log`, `supervisor.log`, per-stage logs.
Single fcntl lock; caffeinate; full atomic epoch checkpoints contain optimizer,
scheduler and Python/NumPy/Torch/MPS RNG. GPU requires escalated project Python.
Use absolute `.venv/bin/python` WITHOUT resolving symlink to system Python.
119 preflight regression tests pass; cohort verification passes; logs saved beside
protocol. Runtime/report/verify/tests follow automatically after full successful run.
No new result is complete yet. Wait autonomously through completion and send meaningful
updates/dataset summaries. If failure, preserve frozen results and fix implementation;
never tune from test outputs or silently restart uncheckpointed task weights.

Source is hash-locked; changes to `cv_hyperparameter.py` require a documented engineering
amendment/repair mechanism before continuing. Do not silently edit after registration.
Protected parent source/hash guards also remain. Helpers/runtime/report can change
without altering model choices. Resume by launching `run_background.py` only after
confirming its supervisor/worker are not alive. No duplicate training processes.

Output root `outputs/results/cv_hyperparameter_v4/<configuration>/fold_i/` and final
under winner configuration; `cv_results.json`, `cv_selection.json`, REPORT, metric
pairs/bootstrap and audit are saved automatically. Runtime helper
`cv_hyperparameter_runtime --activate` must match all139 original-eval sections and
actual CLI examples for NYU/OCSC, then activates CV-chosen single model. It rolls
back workflow on CLI verification failure. Main transcript report should include
v4 audited results first (integration in progress at this handoff write).

## Preserved data, scope and results

Read AGENTS.md. Original isolated/checksummed cohorts are `outputs/validation/cohorts_v1`.
OCSC train211/567sections eval45/125 test45/111; NYU train20/63 eval4/14 test4/15.
Totals train231usable/630, eval49/139, test49/126. Two additional assigned NYU train
children have no eligible sections; two OCSC sources lack selected discourse tasks.
All333 source files inventoried, all895 prepared sections retained. Never reshuffle
original roles, split by section/crop, fabricate labels or drop failures.

Scope transcripts→model/rules→utterances→metrics. Defer audio/MOSS and original ten
participants. Preserve raw files, paper/poster, old weights/results. No researcher
messages or unrelated benchmarks. User hates unsupported leakage statements: say
cohorts are participant-independent, while preserving prior benchmark exposure records.

LUCID lexical benchmark is separate, not segmentation training. Compared against actual
published author TAALED table:180/240rows scored;60vocoder ambiguous. Babble/L2 all15
shared measures within engineering targets; word-countAPE .7648/1.8725%; MATTR .4377/.4389%.
NoBarrier word-count16.5987%, historical exact inputs unresolved. New boundary training
cannot change whole-text lexical values.

Original calibrated test count/MLUw/SD: NYU29.98/16.57/13.70%; OCSC10.36/8.70/14.24%.
Main report `outputs/results/transcript_validation/REPORT.md`. Prior production remains
`outputs/results/benchmark/transcript_workflow.json`, .25turn margin, archived under
`outputs/results/participant_cv_v1/previous_production` with original weights hash.
Parent fixed-recipe5fold complete: NYU16.402/10.574/20.185%, OCSC9.165/9.093/14.885%.
These were231children630sections; not new hyperparameter-selection results.

v2/v3 eval-only ensemble branch preserved: full-train eval11.8071%; five-model12.6844%;
six-model11.5571% at-.25, versus old11.3534%. User corrected CV purpose during v3
scoring, before fresh test metric result saved. V3 test preparation/emission caching
had begun AFTER its frozen choice; do not falsely say test never opened. No test
metric outcomes were used to design v4; that partial branch and history stay preserved.
Do NOT resume old participant_improvement[_refined] run/finish scripts.

Useful measured NYU explanation under `outputs/results/participant_improvement_v3/NYU_EXPLANATION.md`:
train NYU5336words/20children vs OCSC229378/211; pooled train+eval median observed
turn2vs5words, short turns70%vs41%, unintelligibility79.49vs5.73per1000words;
training masked nonfinal gaps11.80%vs1.08%. Native FREQ/MLU word-total gap7.22%vs1.79%.
These are descriptive hypotheses, not causal proof; CV boundaryF1NYU88.81actually exceeds
OCSC87.22, so distinguish boundary F1 from participant count/MLUAPE. Source profile script
and JSON saved separately. Diagram v3 describes obsolete ensemble branch; update to v4.

Earlier detailed state: `outputs/validation/cv_hyperparameter_v4/previous_session_handoff.md`.

## Progress checkpoint during active v4 run

Baseline lr2_count1 all5 folds complete: pooled CV13.1602309003%, fold SD2.4776812833.
Lower count coefficient lr2_count025 all5 complete: pooled13.3566326867%, fold SD2.6818970711.
NYU count/MLUw/SD17.4862/10.9984/19.4203%; OCSC8.8362/8.6674/14.7314%.
Saved lower_count_cv_progress.json. No recipe selected yet. Higher LR/count1
fold1 heldout9.0557084568%, fold2 12.9364647107%; fold3 now training. Follow live
status and logs, not this snapshot. Reporting helper cv_hyperparameter_comparison
now runs automatically from transcript_report only after the full final audit;
it compares frozen test predictions with previous production, retains regressions
and missingness, checks identical words/references/lexical values, and saves paired
participant-bootstrap changes. LUCID_STATUS.md saves the qualified published comparison.
