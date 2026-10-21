# Original coded CHAT reference sensitivity

Native CLAN scores original professional main-tier codes for exactly the same public task sections. This preserves retraces, fillers, replacements, exclusions and continuations. Word MLU uses `-t%mor`. FREQ and clinical surface-word eligibility differ, so those differences are reported as policy sensitivity. No model or normalized-benchmark endpoint was changed.

| Condition | Corpus | Eligible-count APE % | MLUw APE % | SD APE % | FREQ TNW agreement APE % | FREQ NDW agreement APE % |
|---|---|---:|---:|---:|---:|---:|
| normalized_gold | ocsc | 3.32 | 3.79 | 5.10 | 3.83 | 2.78 |
| normalized_gold | nyu_emerson | 3.12 | 3.50 | 3.38 | 10.65 | 11.63 |
| rules | ocsc | 66.06 | 192.48 | 278.98 | 13.66 | 10.85 |
| rules | nyu_emerson | 73.63 | 173.43 | 265.05 | 39.05 | 34.52 |
| previous_minilm | ocsc | 9.31 | 8.83 | 15.46 | 3.40 | 2.63 |
| previous_minilm | nyu_emerson | 34.34 | 18.56 | 19.75 | 4.58 | 5.05 |
| latest_minilm | ocsc | 9.75 | 8.74 | 15.76 | 3.52 | 2.68 |
| latest_minilm | nyu_emerson | 26.43 | 17.23 | 17.60 | 5.35 | 5.94 |
| batchalign_0.10.4 | ocsc | 52.72 | 120.94 | 128.49 | 8.46 | 6.10 |
| batchalign_0.10.4 | nyu_emerson | 61.31 | 111.11 | 81.82 | 29.88 | 24.46 |

`normalized_gold` isolates the change caused by removing CHAT coding and applying our surface profile, while retaining human boundaries. Its nonzero error is a reference-processing difference before segmentation errors. Native MLU eligible utterance count is not total utterance count. CLAN reference ratios/SD are rounded to three decimals. All reference files, commands and outputs are in `original_chat_sensitivity.json`; every metric pair and missing result is retained in CSV.
