# LUCID acquisition and published comparison

Run from the workspace root with the project's `.venv/bin/python`. Existing frozen protocols and completed scores refuse replacement. Do not remove these guards or alter source matching to improve the reported numbers.

## Retained evidence

- `dataset/public/lucid_published/TAALED_results.csv`: unchanged published per-sample reference, with source URLs/SHA-256 in `source.json`.
- `dataset/public/lucid_published/source_docs/`: downloaded citation, filename conventions and TextGrid README.
- `dataset/public/lucid_published/textgrids/combined_manifest.json`: all eight successful archives, 300 extracted TextGrids, about 20 MB fetched. Each `source.json` retains exact ranges, archive ETag, ZIP-entry CRC and extracted-file SHA-256. No complete ZIP hash is claimed. The older `manifest.json` covers only the six split archives.
- `protocol.json`, `prepared.json`: frozen source join, all 240 reference rows, target-tier selection and text hashes. Generic preparation repairs were registered before scoring; initial inputs and amendment are retained.
- `outputs/results/lucid_validation/`: all results, 7,200 method/metric/reference pairs including unavailable rows, summaries and coverage. The 180 scored rows are metadata joined; 60 vocoder rows have two candidates and stay unavailable.
- `osf_listing.json`: public supplement listing. Original cleaned index files and an exact source-to-index mapping were not present.
- `source_audit.json`: article Table 1 averages and published TTR/count consistency. These are provenance/arithmetic checks, not pipeline validation. Large count differences and zero exact token-count matches are retained.
- `audio_download_protocol.json` and `download_first_audio.py`: selection and acquisition of the first published sample's full stereo recording. Its ZIP CRC, length, WAV properties and local SHA-256 are in `dataset/public/lucid_published/audio/LCD_F14BEF3B4cln2.source.json`. No LUCID MOSS inference has run.

## Refresh reports without rerunning inference

```sh
.venv/bin/python -m speech_pipeline.research.lucid_validation report
.venv/bin/python -m outputs.validation.lucid_validation.source_audit
```

The completed original sequence was `prepare`, then `evaluate`, using `speech_pipeline.research.lucid_validation`. To reproduce in a separate copy, retain source data and pin the versions recorded in `outputs/results/lucid_validation/results.json`. The code intentionally rejects existing registered/completed output paths. Preserve original evidence and use an isolated copy for a true rerun.

Acquisition used `speech_pipeline.research.lucid_download` and the two observed archive inventories in `outputs/validation/common_benchmark/`. Large original L2/vocoder ZIPs failed ZIP-header validation, so the publisher's listed split packages were used. The user completed the SpeechBox form; HTTP requests used its exposed sharing link to obtain a server-issued anonymous session. No browser cookies or credentials were read. Browser download-event waits hung and should not be repeated.

## Interpretation

The custom and current TAALED implementations receive identical prepared target-speaker source text. Their differences from 2021 reference numbers can include cleaning, source/tier mapping and software versions; do not call the experiment exact same-input reproduction. The public article describes pause/noise removal, but exact cleaned files and GUI configuration remain unavailable. The production normalizer was not tuned to the results.

TextGrid intervals here are words, not human CHAT utterances. This experiment does not establish ASR accuracy, segmentation accuracy, MLU validity, clinical diagnostic validity or full-pipeline superiority. Scope the healthy-adult dataset to lexical measurement work. `AUTHOR_QUERY_DRAFT.md` contains concrete unresolved questions and has not been sent.
