# User requested stopping

Do not resume training without a new user request. `user_stop.json` prevents the runner from starting. Current status is `stopped_by_user`. Three complete five-fold comparisons and the unfinished fourth configuration’s epoch4 checkpoint are preserved. See `stopped_search_summary.json`, `stopped_search_integrity_audit.json` and the results report. The previous active resume instructions below are historical.

# Continue the CV hyperparameter search

Read job_status.json and SESSION_HANDOFF.md. Do not start a duplicate supervisor or worker.
The single detached runner holds run.lock and writes heartbeat/progress every10seconds.
It completes all4recipes x5folds, freezes CV recipe, fits a fresh final model, freezes
its eval-selected checkpoint/margin, then scores original test. Runtime/report/data
checks/regression tests follow automatically. Log files correspond to each stage.

A running study uses MPS outside the ordinary sandbox. After confirming no saved
supervisor/worker is alive, resume using the absolute project `.venv/bin/python`
(without resolving its symlink) and this directory's `run_background.py` under
`caffeinate -i`. Full epoch checkpoint/optimizer/scheduler/Python/NumPy/Torch/MPS RNG
state allows continuation on the same hardware. Completed selections/scores are
preserved; never delete them to make a run resume. A failed audit is not permission
to alter model/reference choices.

Protected source/protocol hashes must match. If an implementation repair is needed,
preserve the protocol and record a clear amendment; do not silently bypass its guard.
Original cohort exports and raw CHAT files must remain unchanged. CV recipe ranking
uses only original train; original eval/test remain outside the search. Final test
outcomes do not choose a recipe, margin or deployment. No further grid expansion
from partial CV or test results. LUCID remains a separate lexical benchmark.

Previous v2/v3 ensemble workers were stopped and superseded. Do not restart them.
V3 had begun frozen-choice test preparation/emission caching, but produced no fresh
metric result. Preserve that history and all prior results; do not claim tests were
never opened or reshuffle original cohort assignments.

Engineering launch optimization: `clan_launch_amendment.json` records the
corpora.py CLAN process-start change, old/new source hashes and exact native-value
parity on all139 original-eval sections. All119 tests pass. The first worker was
intentionally stopped after its complete lower-count fold2 epoch6 checkpoint, then
resumed with optimizer/scheduler/RNG intact; incomplete calibration was repeated.
Model source, grid and registered protocol remain unchanged. Old run status is
preserved in `job_status_before_spawn_restart.json`.
