# Direct published LUCID/TAALED comparison

Reference: Baese-Berk et al. (2021), DOI [10.3389/fpsyg.2021.661415](https://doi.org/10.3389/fpsyg.2021.661415), actual per-sample OSF TAALED output. Corpus TextGrids were obtained through the authorized SpeechBox data-use link and verified by ZIP-entry CRC and extracted SHA-256. Both current implementations receive the same prepared target-speaker words. No model or normalizer was tuned.

All 240 published rows retained. Coverage: **{'scored': 180, 'ambiguous_source_recording': 60}**. Ambiguous/missing source rows are unavailable, not zero-error or silently dropped.

| Method | Condition | Scored rows | Token-count APE % | Type-count APE % | TTR APE % | MATTR50 APE % | HD-D APE % | Original MTLD APE % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| custom | noBarrier | 60 | 22.84 | 19.96 | 8.65 | 3.03 | 2.40 | 14.67 |
| custom | vocoder | 0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| custom | babble | 60 | 9.78 | 12.77 | 3.87 | 2.29 | 1.37 | 7.51 |
| custom | L2 | 60 | 9.63 | 12.34 | 4.24 | 1.66 | 0.96 | 7.21 |
| taaled_current | noBarrier | 60 | 22.98 | 19.69 | 8.58 | 3.01 | 2.38 | 14.01 |
| taaled_current | vocoder | 0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| taaled_current | babble | 60 | 8.93 | 11.90 | 3.97 | 2.08 | 1.34 | 6.71 |
| taaled_current | L2 | 60 | 9.67 | 12.25 | 3.95 | 1.63 | 0.96 | 7.42 |

APE is mean recording APE within speaker, then mean across speakers, for each condition. The full table retains all 15 shared metrics, signed errors, exact-precision match counts and missingness. The original MTLD and published moving-average MTLD variants are distinct; no unsupported field mapping was made.

## Interpretation

These are comparisons joined by documented speaker/condition/scene metadata; the exact historical cleaned inputs have not been recovered. They test oracle-transcript lexical measurement, not ASR, clinical diagnosis, human CHAT segmentation or whole-pipeline superiority. The original authors removed filled/unfilled pauses and vocal noises, but did not release the exact cleaned indexN.txt inputs or the full historical tool configuration here. Differences may therefore include source mapping, preprocessing, POS/lemmatization and software-version drift. They must not be described as exact reproduction or fixed by choosing favorable recordings. Vocoder has multiple source recordings per speaker/scene key and requires an author-supported mapping before a valid single-row comparison. No scored sample reproduces the published token count exactly with either current implementation. Large no-barrier count discrepancies also require source/tier mapping review, beyond assuming software drift. The table is a discrepancy audit, not evidence of exact reproduction or validated accuracy.

Source TextGrids are word intervals, not professional CHAT utterance gold. No MLU or segmentation score is inferred from their row count. Reference, source, prepared-input and software hashes and the frozen protocol are in `outputs/validation/lucid_validation/`.
