# OCSC audio download complete

All **301 eligible Ohio Child Speech Corpus recordings** are downloaded and verified.
Total: **54,991,744,472 bytes (51.215 GiB)**. Original audio formats are preserved:
298 mono 44.1 kHz WAVs and 3 stereo 48 kHz WAVs. No resampling has occurred.

| Frozen cohort | Children / recordings | Original selected sections | Audio directory |
|---|---:|---:|---|
| Train | 211 | 567 | `dataset/public/ocsc/audio/train/` |
| Validation | 45 | 125 | `dataset/public/ocsc/audio/eval/` |
| Test | 45 | 111 | `dataset/public/ocsc/audio/test/` |
| Total | 301 | 803 | `dataset/public/ocsc/audio/` |

Every recording retains its original utterance-identification cohort. The final
audit checked complete local SHA256, byte count, matching raw CHAT provenance,
WAV container/final-frame structure, and physical cohort placement. It found
**zero missing recordings, zero duplicate audio files across participants, zero
unexpected audio paths and zero remaining partial transfers**. Each WAV has a
neighboring `.source.json` provenance record. Full recordings retain additional
source tasks beyond the 803 selected discourse sections; future training and
scoring must use the registered section inventory.

The frozen cohort bundle still passes verification: 895 sections, 329 eligible
children, zero participant overlap. Two OCSC transcripts without selected
discourse tasks remain excluded; they have not been assigned a new cohort.

Evidence: `ocsc_verification.json`, `ocsc_verification.log`, `download_status.json`,
`download.log`, browser-import receipts and `downloader_amendment.json` in this
folder. Earlier rejected/interrupted attempts remain separately archived.
The final downloader run ended successfully. All **158 regression tests pass**.

Original transcripts, frozen assignments, stopped experiments, result tables and
production weights remain preserved. No new MOSS inference, fine-tuning, scoring
or model activation has run. NYU–Emerson media still requires additional account
access; the supplied account received an explicit authorization denial. The new
MOSS study remains pending audited window preparation, training/evaluation runner
implementation and source access. Existing test exposure remains documented.
