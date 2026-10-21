# First MOSS public-audio pilot

The first frozen public test task was selected before generation: OCSC 4002, ExpPictures, 300.021 seconds. MOSS received audio and its official default prompt only. The existing latest MiniLM then segmented its unpunctuated target-speaker words. No model was changed.

Oracle target speaker: **S02**. Target WER: **18.82%**; speaker-permutation cpWER: **14.29%**.

Speaker roles were assigned using the gold-reference minimum-edit permutation for evaluation. This does not validate automatic patient identification. The MOSS speaker IDs themselves were predicted from audio.

| Method | Metric | Original CHAT/native CLAN | Predicted | Absolute error % |
|---|---|---:|---:|---:|
| latest_minilm | eligible_mlu_utterances | 72.000 | 70.000 | 2.778 |
| latest_minilm | MLUw | 2.486 | 2.457 | 1.161 |
| latest_minilm | length_sd | 1.143 | 0.951 | 16.756 |
| latest_minilm | mlu_words | 179.000 | 172.000 | 3.911 |
| latest_minilm | freq/TNW | 179.000 | 172.000 | 3.911 |
| latest_minilm | freq/NDW | 73.000 | 70.000 | 4.110 |
| latest_minilm | freq/TTR | 0.408 | 0.407 | 0.251 |
| latest_minilm | freq/MATTR | 0.538 | 0.539 | 0.221 |
| moss_source_segments | eligible_mlu_utterances | 72.000 | 61.000 | 15.278 |
| moss_source_segments | MLUw | 2.486 | 2.820 | 13.422 |
| moss_source_segments | length_sd | 1.143 | 1.797 | 57.232 |
| moss_source_segments | mlu_words | 179.000 | 172.000 | 3.911 |
| moss_source_segments | freq/TNW | 179.000 | 172.000 | 3.911 |
| moss_source_segments | freq/NDW | 73.000 | 70.000 | 4.110 |
| moss_source_segments | freq/TTR | 0.408 | 0.407 | 0.251 |
| moss_source_segments | freq/MATTR | 0.538 | 0.539 | 0.221 |
| latest_minilm_native_clan | eligible_mlu_utterances | 72.000 | 70.000 | 2.778 |
| latest_minilm_native_clan | MLUw | 2.486 | 2.486 | 0.000 |
| latest_minilm_native_clan | length_sd | 1.143 | 0.937 | 18.023 |
| latest_minilm_native_clan | mlu_words | 179.000 | 174.000 | 2.793 |
| latest_minilm_native_clan | freq/TNW | 179.000 | 174.000 | 2.793 |
| latest_minilm_native_clan | freq/NDW | 73.000 | 71.000 | 2.740 |
| latest_minilm_native_clan | freq/TTR | 0.408 | 0.408 | 0.000 |
| latest_minilm_native_clan | freq/MATTR | 0.538 | 0.541 | 0.558 |
| moss_source_segments_native_clan | eligible_mlu_utterances | 72.000 | 61.000 | 15.278 |
| moss_source_segments_native_clan | MLUw | 2.486 | 2.852 | 14.722 |
| moss_source_segments_native_clan | length_sd | 1.143 | 1.773 | 55.118 |
| moss_source_segments_native_clan | mlu_words | 179.000 | 174.000 | 2.793 |
| moss_source_segments_native_clan | freq/TNW | 179.000 | 174.000 | 2.793 |
| moss_source_segments_native_clan | freq/NDW | 73.000 | 71.000 | 2.740 |
| moss_source_segments_native_clan | freq/TTR | 0.408 | 0.408 | 0.000 |
| moss_source_segments_native_clan | freq/MATTR | 0.538 | 0.541 | 0.558 |

Native references preserve original professional CHAT coding. Count means eligible MLU utterances. Methods suffixed `_native_clan` use identical native MLU/FREQ programs on full recognized-word CHAT; other rows use the normalized clinical surface profile. FREQ vs clinical surface-word differences include coding/eligibility differences. MOSS source segments have audio context; they are a different-input baseline from the paragraph model. No word-gap boundary F1 is reported on different ASR word streams. The faithful provisional CHAT and separate normalized calculator input are in `latest_minilm/`. No precise utterance timestamps, MOR, grammar, maze or production-error labels were fabricated.

**This is one feasibility pilot, not a validated cohort result or an exact published-result replication.** Next: freeze a multi-participant audio manifest, specify reviewed target mapping and common word/CHAT eligibility, then compare identical recordings with an established audio system and published reference metrics.
