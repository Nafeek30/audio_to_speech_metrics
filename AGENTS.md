# Resumed CV completion & external benchmark evaluation — 20 October 2026

The user explicitly requested:
1. Finish the 5-fold cross-validation on the original training datasets (`cv_hyperparameter_v4`), completing all configurations, fitting the fresh final model on original train, selecting epoch/margin on eval, scoring original test, and deploying the winning model.
2. Re-run all external dataset benchmarks (ENNI, OCSC test, LUCID) with the new model.
3. CRITICAL MANDATE: Never train or validate on external datasets (ENNI, DementiaBank, etc.). Keep them strictly held-out, zero-shot external test benchmarks only.

Preserve all raw sources, cohort boundaries, and prior experimental logs.

# Earlier Handoff and Pause (Historical Reference)

For a future resumed MOSS study, the user chose OCSC ONLY and deferred NYU audio.
All 301 OCSC recordings are verified in original 211/45/45 child cohorts. Preserve
NYU data/results and all original assignments. The user permits target-cluster
identification from at most the first three reference sentences; the proposed
initial-child-utterance mapper remains unimplemented. Define the inference unit
and sentence interpretation before running it, freeze the mapping, exclude the
calibration prefix from primary scores, and never match against later reference
text or rewrite ASR words. Report this as reference-assisted identification.
The historical sealed two-source registration remains unchanged; a separate
OCSC-only scope and prefix-policy amendment is required before future training.
Preserve the old MiniLM stop guard, all results and production weights.

## Earlier MOSS authorization — subsequently paused above

On October 20 the user explicitly requested matching OCSC and NYU–Emerson audio,
the SAME frozen train/validation/test cohorts, human CHAT supervision, and NEW
MOSS-Transcribe-Diarize fine-tuning with participant-grouped CV. This authorizes
the separate audio study, acquisition, runtime setup, training and scoring after
data/access/label checks. It does NOT resume the old MiniLM experiment or authorize
production activation. Keep cv_hyperparameter_v4/user_stop.json and all old results.
New registration: outputs/validation/moss_audio_v1/registration/protocol.json.
Read the latest SESSION_HANDOFF.md entry for acquisition/access status. Preserve
source transcripts; reference text is permitted only for the explicitly authorized
initial target-cluster matching described above, not ASR prompts or word correction.
Existing test exposure remains explicit. No researcher contact.

## Historical stopping point and acquisition-only authorization

New authorized scope: the user requested audio-download scripts for all three
public sources. Acquisition into separate source/cohort folders is allowed.
Do not restart training, ASR scoring or model selection. Use the frozen cohort
manifest and preserve source transcripts. Scripts and access instructions are
`scripts/download_public_audio.py` and `scripts/README_AUDIO_DOWNLOAD.md`.

On October16 the user explicitly requested stopping at the best completed results.
Read SESSION_HANDOFF.md and cv_hyperparameter_v4/user_stop.json. Do not restart
training, finish the fourth recipe, run a fresh final test or activate new weights
without a new user request. The runner has a user-stop guard. Current task is closed.

- Three configurations completed all five original-training participant folds.
  Baseline lr2_count1 has the best combined CV tuning score13.1602%; lower count
  weight has the best OCSC three-metric CV mean. OCSC improved from the earlier
  fixed-recipe9.16/9.09/14.89% to8.84/8.67/14.73% count/MLUw/SD. Report the tradeoff:
  lower count weight worsened NYU, so it did not improve the combined objective.
- Fourth configuration lr3_count025 stopped during fold1, with complete epoch4
  checkpoint saved, no frozen fold selection or heldout score. Exclude this whole
  configuration from ranking. Do not claim the registered four-setting study finished.
- Existing production workflow/weights remain preserved and active. No v4 final fit,
  final test or activation occurred. CV tuning values are distinct from existing
  production-test values; do not combine them or cherry-pick metrics across models.
- Original train/eval/test cohorts are participant-independent and immutable.
  Every section/crop follows its child. All15 completed CV assessments passed the
  stopped-search integrity audit, including fit-only crops and native word/metric
  checks. Prior benchmark exposure remains documented; do not claim newly unexposed
  external confirmation or repeat blanket claims of cohort overlap.
- LUCID lexical-only agreement is with actual published TAALED values. Babble/L2
  all15 shared mean errors meet targets; noBarrier and ambiguous vocoder remain
  qualified. No boundary-training examples come from LUCID.
- NYU explanation is descriptive: younger median age4.58 versus7.22years in
  train/eval, shorter turns, more unintelligibility and less training information.
  Do not claim age was proved to cause the error or that every NYU metric is worse.
- Preserve raw sources, cohort exports/manifests, all models/checkpoints/results,
  paper/poster and the stopped experiment history. Audio/original10participants
  remain deferred. Do not contact researchers or create unrelated work.
- Native CLAN launch-only optimization is documented in clan_launch_amendment.json:
  all139 eval reference values matched and119 tests passed. Model/protocol sources
  remain hash-locked. Repairs must be documented; never silently bypass guards.
