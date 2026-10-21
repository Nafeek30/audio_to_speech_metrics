# Continue the registered fresh five-fold study

The user requested participant-grouped k=5 cross-validation, authorized fresh task-model training, and explicitly asked that the job continue/save independently of chat token limits. Original train/eval/test assignments and source data remain unchanged.

## Check status before doing anything

```sh
cat outputs/validation/participant_cv_v1/job_status.json
tail -n 30 outputs/validation/participant_cv_v1/training.log
```

- `waiting_for_existing_worker`: the detached supervisor is preserving the first foreground fold until its selection is frozen. It then transfers to the checkpoint-capable runner. `foreground_initial.log` preserves the initial worker's output. Its expected implementation-hash stop at the transfer boundary is covered by `implementation_amendments.json`; this changes runtime durability, not the experiment.
- `running`: do not start another training process. The heartbeat and per-fold progress update every 10 seconds. `run.lock` prevents duplicate supervisors.
- `complete`: the supervisor checked that all five outer-fold results, the separate final-model test results, the pooled cross-validation JSON and the report exist. Verify results and integrity before deploying anything.
- `failed` or `supervisor_failed`: inspect the saved error/log. Completed frozen folds are reused; unfinished jobs resume from their own epoch checkpoints. Do not change cohorts, hyperparameters, seeds or selection rules in response to held-out results.

The detached launcher is recorded in `launcher.json`. It uses `caffeinate -i` to prevent system idle sleep while running, with separate sessions/stdin detached from chat. A shutdown/reboot still requires resuming the job; checkpoints remain on disk.

Transfer verified: fold 1 finished and froze naturally, then its outer evaluation completed in the detached worker. No partial fold restart was necessary. Fold 2 started on MPS and its epoch-zero full-state checkpoint was verified on disk. The detached worker PID at verification was 35547; use the live status file rather than assuming that PID remains current.

## Resuming after a stopped process

First ensure the recorded supervisor/worker PIDs are not still running. Launch the same supervisor with no legacy PID:

```sh
.venv/bin/python outputs/validation/participant_cv_v1/run_background.py
```

For unattended continuation, launch this command with `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL, stdout=supervisor.log, stderr=STDOUT)` under `/usr/bin/caffeinate -i`. In Codex, use the already authorized `.venv/bin/python` command with escalated GPU access: the restricted sandbox reports MPS unavailable. A saved MPS checkpoint deliberately refuses CPU continuation to avoid silently changing the execution device.

Each unfinished training job saves atomic `training_state.pt`: latest model, AdamW state, scheduler state, epoch/history/best/stale values, Python/NumPy/Torch/MPS RNG states and protocol hash. Checkpoints begin at epoch zero and are updated after every completed epoch. `checkpoint_status.json` records the latest complete epoch. The best selected inference bundle is separately saved under `model/`; selection is frozen in `selection.json` before outer evaluation. Interrupted writes leave the previous checkpoint usable. Epoch-boundary resume may repeat an interrupted partial epoch; it does not reuse another fold's weights.

All five original pretrained files are backed up durably under `pretrained_original/` and verified against the immutable protocol hashes. The training code uses this byte-identical fallback if the registered temporary cache disappears. Remaining folds do not require that temporary cache to survive a reboot.

The original foreground worker predates optimizer checkpoints. If it finishes fold 1 normally, its frozen model/selection are sufficient and training is not repeated. If it ends before freezing, the supervisor preserves its incomplete folder under `legacy_interrupted_fold_1_*` and restarts the same registered fold/seed from the original pretrained initialization, logging the reason. It never promotes partial weights into held-out scoring.

## What the run must produce

- `outputs/results/participant_cv_v1/fold_1/` through `fold_5/`: fresh initialization proof, training history, selected model, inner calibration, frozen selection and outer results including utterances/native CLAN outputs/boundaries.
- `outputs/results/participant_cv_v1/cross_validation_results.json`: all 630 original training sections, exactly one out-of-fold prediction each; pooled participant metrics and five-fold variability, separate by corpus.
- `outputs/results/participant_cv_v1/final/`: a separate fresh model trained on all original train; original eval selects epoch/margin; original test opens only after selection is frozen.
- `outputs/results/participant_cv_v1/REPORT.md`: CV results and separate final-model test results.
- `outputs/results/participant_cv_v1/previous_production/`: unchanged old weights/workflow archive. Current production has not been replaced or deleted.

After completion, run the cohort verifier and regression suite; audit every fold's fit/inner/outer identities, initialization hashes, exactly-once outer coverage and final freeze-before-test chronology. Confirm model/word/reference integrity and production-command parity before changing deployment. Never deploy the best outer fold or choose from test results. Update the main report, README and SESSION_HANDOFF with actual completed results.

## Fixed design and interpretation

The immutable protocol is `protocol.json`; durability-only code amendments are in `implementation_amendments.json`. Five outer folds use only original training participants (231 usable children, 630 sections); the original 49 eval and 49 test children remain outside CV. Each fold holds out a distinct outer participant group and uses a separate inner validation group for epoch/calibration selection. Fresh Microsoft MiniLM pretrained initialization is used for every fold and the final model. LUCID remains a separate lexical benchmark.

Previous runs used one participant-disjoint split, with recorded model selection before their test run; no completed k-fold study existed. Do not conflate participant independence with a new external corpus or erase the recorded earlier benchmark inspection. This study supplies participant-independent out-of-fold internal validation of the fixed recipe.

104 regression tests passed before detached launch, including a checkpoint round-trip that reproduces the next optimizer update and Python/NumPy/Torch random draws exactly after restoration. GPU checkpoints additionally retain MPS RNG state.
