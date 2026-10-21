# Latest continuation — prospective calibration refinement before fresh test

All five CV jobs and the full-training job are complete; no fresh final test has been opened.
Original train/eval/test cohorts remain participant-independent. All six encoders initialized
from original pinned weights with fresh distinct heads; no old fine-tuned weights loaded.
CV pooled count/MLUw/SD APE: NYU16.402/10.574/20.185%, OCSC9.165/9.093/14.885%.
Full-train fresh model selected epoch3, margin0, evalobjective11.8071%, versus prior11.3534%.
Its positive margins worsened eval sharply, so user goal of best results motivated a bounded
**pretest** calibration refinement: add -1,-.5,-.25,-.125,+.125 to the seven original margins,
for 12 margins x3 fixed model candidates (36). This uses eval only; weights/splits unchanged.
Original v2 protocol, completed comparisons and six caches preserved. Test was explicitly
absent at refinement registration. New results/evidence under `participant_improvement_v3`.
New module: `speech_pipeline.research.participant_improvement_refined`, runtime and figures
likewise `_refined_runtime`, `_refined_figures`. Main supervisor resumes all complete trained
models, reuses validated v2 comparison entries/caches, recomputes selected native eval result,
freezes v3 winner, then tests exactly that winner. No further grid expansion after this.
Deployment still requires beating prior11.3534% evalobjective; test does not decide activation.
Read parent job_status/training.log and v3 finish_status. Separate v3 finisher handles runtime
parity/guarded activation, figure/main report, cohort verification and tests. Main report now
reads v3. Supersedes v2-only runtime/output instructions below. Keep waiting through completion.

# Current task — accuracy improvement with intact cohorts

Updated October 15–16, 2026. Read live status files; the snapshots below are not completion claims.

## User goal and authorized scope

Improve transcript-based metric/utterance results on LUCID, NYU–Emerson and OCSC.
User explicitly authorized fresh training, participant-grouped k=5 CV, model improvement
using CV, and retiring previous weights if warranted. Continue to finished results now;
do not stop at a background-run status. Give dataset summaries and honest regressions.
Defer audio/MOSS and the original ten participants. Preserve source data, cohorts,
paper/poster and prior results. No researcher messages or unrelated benchmarks.

## Cohorts

Train, eval and test are **participant-independent**. Do not imply leakage was found.
Original isolated, checksummed exports: `outputs/validation/cohorts_v1/`.
Usable counts: OCSC train211/567 sections, eval45/125, test45/111;
NYU train20/63, eval4/14, test4/15. Totals train231/630, eval49/139, test49/126.
All 895 sections and 333 source CHAT files audited. Two extra NYU training children
lack eligible sections; two OCSC sources contain no selected discourse task.
Never move participants between original roles or fit normalization/vocabulary/crops
on eval/test. `load_split` requires one explicit role; training crop provenance checked.
Historical benchmark exposure is preserved in protocols; it is different from participant
leakage. Describe cohort separation precisely, without blanket claims of invalid results.

## Live work — read these first

- `outputs/validation/participant_cv_v1/job_status.json`: main supervisor/worker PIDs,
  heartbeat, epochs/checkpoints, frozen selections and completed held-out scoring.
- `outputs/validation/participant_cv_v1/training.log`: live training/calibration/results.
- `outputs/validation/participant_improvement_v2/finish_status.json`: post-training
  runtime activation guard, figure/report, cohort verification, regression steps.
- `outputs/validation/participant_improvement_v2/CONTINUE.md`: restart instructions.

Main worker: `speech_pipeline.research.participant_improvement run`, MPS, under
caffeinate/detached supervision. Do not start duplicates: inspect PID/command/locks.
Current snapshot: first three folds frozen/evaluated (385 sections); fourth training.
No fresh final test opened. All source words/native references preserved in partial audit.
112 regression tests pass. Completed jobs are preserved. Atomic full-epoch checkpoints
contain optimizer/scheduler/Python/NumPy/Torch/MPS RNG; resume same hardware.
Use absolute `.venv/bin/python` **without resolving its symlink**. System Python lacks
project packages. One failed restart stopped at import before training; corrected.

## Fixed protocols and improvement selection

Parent `outputs/validation/participant_cv_v1/protocol.json`:
five outer folds within original train only, each with an inner participant holdout.
Every model starts from original pinned MiniLM encoder/new task head, never previous
fine-tuned weights. Fixed custom gap classifier, six epochs/patience2, corpus-balanced
sampling, CE+count loss, fit-only crop/UNK augmentation. Inner-only epoch selection,
then seven margins (0,.25,.5,.75,1,1.5,2). Outer scores never choose anything.
A sixth fresh model fits all original train, epoch/margin selected on original eval.
Parent final single-model test is held by `evaluation_hold.json`.

Improvement `outputs/validation/participant_improvement_v2/protocol.json`:
compare full-train single model, equal-probability ensemble of five fold models,
and equal-probability ensemble of all six. Same seven margins, original eval only.
Objective: corpus-balanced mean participant APE across eligible count/MLUw/length SD.
Freeze candidate/margin/component hashes, then score ONLY chosen candidate on original test.
No weights are averaged across randomly initialized heads. No ensemble score on original
train may be labeled OOF: OOF uses its dedicated held-out fold model only.

Registered deployment_policy.json: activate new winner only if its eval objective
strictly improves archived production's 11.35342133442171%; otherwise retain previous.
No test scores drive activation. Both protocols/implementation hashes are checked.
Changes to protected source require explicit prospective amendments, not silent edits.

## Outputs and remaining completion steps

- Parent `outputs/results/participant_cv_v1/`: fold1..5 model/history/selection/outer;
  final fresh full-train model; combined OOF JSON/report/CSV/bootstrap; METHODS.md.
- New `outputs/results/participant_improvement_v2/`: 21 dev candidates, frozen winner,
  selected eval predictions, chosen final-test predictions, audits/CSV/bootstrap/report.
- Supervisor finishes training + dev selection + test + audit/report. A second detached
  finisher waits for successful completion, then runs new runtime `--activate` (guarded),
  scientific figure, combined transcript report, cohort verify and final regression tests.
- Correct new runtime: `speech_pipeline.research.participant_improvement_runtime`.
  Old `participant_cv_runtime` expects obsolete single-model test; do not run it.
- New runtime must match ALL139 eval sections and actual transcript CLI examples for
  each corpus before deployment. `transcript.py` explicitly loads saved model/hash.
- After everything finishes, inspect errors, fix implementation issues without changing
  frozen models/choices, verify final audits and numbers. Update this handoff with actual
  results/production choice; open combined report and deliver concise user summary.
- Combined report: `outputs/results/transcript_validation/REPORT.md`. Generator includes
  new scores only after complete new audit. Preserve all earlier historical comparisons.

## Completed lexical/baseline results

LUCID lexical correction compared directly against the authors' per-sample published
TAALED table (Baese-Berk et al. 2021, DOI10.3389/fpsyg.2021.661415). 180/240 rows scored;
60 vocoder rows ambiguous, not zero-filled. All15 shared metrics meet engineering targets
for babble/L2 (120 samples). Word-count APE .7648%/1.8725%, MATTR .4377%/.4389%.
NoBarrier word-count error16.5987%; historical exact cleaned inputs/model still unresolved.
LUCID supplies no segmentation training data and is unchanged by current boundary training.

Previous calibrated test count/MLUw/SD APE: NYU29.98/16.57/13.70%;
OCSC10.36/8.70/14.24%. Original test cohort all49 children/126 sections.
Previous production archived intact under parent `previous_production`.
Raw/native references and old prediction outputs are preserved, including regressions.
Vocabulary errors remain mostly below3%; unsupported annotation-dependent clinical metrics
stay unavailable. Never invent reference counts/labels, leak human utterance boundaries,
choose recordings by numerical proximity, or drop failures.

More historical detail: `outputs/validation/participant_improvement_v2/session_handoff_previous.md`.
