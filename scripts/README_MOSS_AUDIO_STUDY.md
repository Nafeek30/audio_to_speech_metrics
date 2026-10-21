# MOSS audio study — original cohorts retained

**Paused for handoff, October 20:** the user asked to stop work for now and save
notes. Read [the consolidated handoff](../HANDOFF_2026-10-20.md). Do not launch
downloads, training, inference or scoring until the user asks to resume. The stopped
MiniLM experiments and their stop marker remain preserved. MOSS training, model
selection and new scoring have not run. Do not describe setup or the existing
single-recording pilot as a completed fine-tuning experiment.

OCSC acquisition is now complete: all 301 eligible recordings (51.215 GiB), with
211/45/45 original train/validation/test cohorts and a passing final integrity
audit. See `outputs/validation/moss_audio_v1/acquisition/OCSC_DOWNLOAD_COMPLETE.md`.
The user chose **OCSC-only** future audio training and deferred NYU–Emerson.
MOSS fine-tuning remains pending. No acquisition or training worker is running.

## Frozen eligible participants and source recordings

| Source | Train | Validation (`eval/`) | Test | Total |
|---|---:|---:|---:|---:|
| Ohio Child Speech Corpus | 211 | 45 | 45 | 301 |
| NYU–Emerson | 20 | 4 | 4 | 28 |

All 895 original selected sections retain their child and original cohort. Two
additional assigned NYU training children have no eligible sections, and two
OCSC transcripts have no selected discourse tasks; do not add them to this study.
LUCID is outside these audio-training cohorts.

The registration is
`outputs/validation/moss_audio_v1/registration/protocol.json`, with a neighboring
SHA256 seal. This is the historical **two-source** registration: preserve it and
write a separate OCSC-only scope/prefix-policy amendment before future experiments.
It binds the original cohort manifest and existing five participant
folds. CV remains entirely within original training participants. Original
validation and test participants never enter a gradient update. Every recording,
section and eventual crop follows its original participant. Test data and the
OCSC 4002 pilot are historically exposed; do not claim a pristine new test cohort.

## Acquisition and final verification

Read live progress in
`outputs/validation/moss_audio_v1/acquisition/download_status.json` and
`download.log`. Browser imports have separate receipts/status; the bulk downloader
verifies and skips those existing files. Failed/interrupted attempts are retained
as `.attempt1` and `.attempt2`. The downloader repairs and their evidence are
documented in `downloader_amendment.json`.

The signed-in account has Ohio access. NYU–Emerson currently shows **Not
authorized**, also confirmed by its authenticated direct media response. Additional
source account access is required; do not circumvent the restriction or contact
researchers on the user's behalf. Credential values stay outside the repository.

Before declaring OCSC complete, run `scripts/verify_public_audio.py --sources
ocsc --report outputs/validation/moss_audio_v1/acquisition/ocsc_verification.json`.
Require all 301 recordings, 211/45/45 original cohorts, valid full local SHA256,
matching raw transcript provenance, WAV structure, no missing files, and no
identical recordings assigned to different children. Do not infer completion
from folder creation or a running process.

## Label audit and runtime

Separate `train/`, `eval/`, and `test/` reference-only JSONL files preserve all
speakers, raw CHAT main tiers, audible words, supplied timing and explicit quality
flags. They are not directly consumable training examples. Original CHAT files,
including dependent tiers and source processing comments, remain untouched.
Source CHAT comments identify automatic alignment and morphology processing, so
do not label those fields as manually verified human time/morphology annotations.

Before exporting audio windows, resolve or explicitly exclude training spans with
missing/invalid timing, unintelligibility, anonymized speech and unsupported
nonlexical events. Preserve coverage and disclose unavailable evaluation scores;
do not silently select easy validation/test excerpts. Gold roles, text, time labels
and boundaries are supervision/reference only, never inference prompt features.
The user separately permits target-cluster identification by matching generated
text against no more than the first three reference sentences. The proposed
initial-child-utterance mapper is not implemented; define its inference unit and
sentence interpretation, freeze the cluster choice, exclude the calibration prefix
from primary scores, and report reference-assisted identification. Do not consult
later reference words or rewrite recognized text. The old full-reference oracle
pilot does not meet this new restriction.
Training should cover all audible speakers with anonymous local speaker IDs.
An oracle permutation for cpWER scoring does not validate automatic child-role
identification.

Separate runtime: `outputs/runtime/moss_audio_v1/venv`.
Official source revision: `e3bdd93f8ad58418e04659818efd5fd87a0bc486`.
Base model revision: `704aa4a9c304e8520be88901e0d1960158ef5b15`.
Official [fine-tuning guide](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md).
Dependencies are frozen in `outputs/validation/moss_audio_v1/runtime_requirements.txt`.
The pinned base and processor load successfully: 908,513,280 parameters, 16 kHz
input frontend. The Mac has 32 GiB memory and MPS is available outside the sandbox;
the CPU load check in `runtime_preflight.json` ran inside it. No gradient update,
new inference, ASR scoring or model activation has occurred.

Remaining work if the user resumes: the OCSC-only/prefix protocol amendment,
audited audio-window construction,
participant-safe training and checkpoint selection, meaningful WER/cpWER and
native CHAT measure evaluation, and a final frozen descriptive test. A complete
training/evaluation runner has not yet been implemented or launched.
