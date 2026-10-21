# Conference validation status — public data

The ten-participant cohort is excluded from this continuation. Frozen model weights and selection are unchanged. The work now includes a matched public competitor comparison, native original-CHAT reference checks, a real MOSS audio pilot, and a direct published LUCID discrepancy audit.

## Evidence completed

**Matched transcript comparison:** 49 public test children, 126 paragraphs, 54,654 words. Every method receives the same target-speaker words without gold punctuation or turn/utterance cuts. Latest MiniLM's corpus-balanced count/MLUw/population-SD APE is **16.09%**, versus **87.92%** for the official Batchalign 0.10.4 segmentation component. All 504 word streams are preserved. All 30 predeclared native CLAN arithmetic checks passed. An additional full official Batchalign recipe smoke test reproduced the direct backend's word stream and 16 utterances on the first paragraph.

This is a new common-data **component** comparison. Our models were trained on disjoint children from these corpus domains; the official pretrained component was not adapted here. The reused test is not independent confirmation, an equal-adaptation architecture comparison, or reproduction of the adult clinical cohort in [Liu et al. (2023)](https://aphasia.talkbank.org/publications/2023/Liu23.pdf). Full methods, metric pairs, coverage and paired child intervals are in [REPORT.md](REPORT.md).

**Original professional CHAT references:** native CLAN was run on all 126 original coded task sections. Latest MiniLM's MLUw APE is approximately **8.74% OCSC / 17.23% NYU**, eligible-count APE **9.75 / 26.43%**, and length-SD APE **15.76 / 17.60%**. These comparisons expose real coding/eligibility differences that identical-normalized-text calculator checks hide. See [ORIGINAL_CHAT_AUDIT.md](ORIGINAL_CHAT_AUDIT.md).

**First real audio pilot:** the first saved test task, OCSC 4002 ExpPictures, was selected before generation. Pinned MOSS processed the entire 300.021-second excerpt with its official prompt, producing a complete EOS result. The same frozen segmenter then processed its target-speaker words. Native CLAN scored system and professional reference CHAT with identical commands:

| Measure | Professional reference | MOSS + frozen segmenter | Absolute error |
|---|---:|---:|---:|
| Eligible utterances | 72 | 70 | 2.78% |
| MLUw | 2.486111 | 2.485714 | 0.016% |
| Population length SD | 1.143 | 0.937 | 18.02% |
| Total eligible words | 179 | 174 | 2.79% |
| Different words | 73 | 71 | 2.74% |
| TTR | 0.407821 | 0.408046 | 0.055% |
| MATTR50 | 0.538 | 0.541 | 0.56% |

MOSS source segments alone produce 61 eligible units, MLUw 2.852459 and SD 1.773, so the post-segmentation step improves these errors on this sample. Target WER is **18.82%**, all-speaker cpWER **14.29%**. The close MLU reflects cancellation of word/count errors and cannot establish accurate recognition. Speaker-role matching is explicitly **oracle, reference-assisted evaluation**, not automatic patient identification. One sample provides feasibility evidence only. All raw outputs, timing audits, denominator definitions, native outputs and provisional CHAT are in [the pilot report](../moss_pilot/REPORT.md). No boundary F1 was computed on different ASR word streams. Native MLUw/TTR use exact counts; SD/MATTR have three-decimal native precision.

## Exact published-result reproduction

[Baese-Berk et al. (2021)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.661415/full) release actual TAALED results and analysis code for LUCID. The unmodified public supplement has been acquired and inventoried: 240 result rows, 40 speakers, four conditions, unique speaker/condition/scene keys. It contains shared counts, lexical densities, TTR/log-TTR/Maas, MATTR, MSTTR, HD-D and original MTLD; its moving-average MTLD variants must not be silently equated to our mean-factor-length measure.

The user completed the SpeechBox data-use form. **All 300 source TextGrids and one complete stereo audio file are downloaded.** ZIP entry CRCs, extracted SHA-256 hashes and archive ETags are retained. Selective transcript acquisition fetched about 20 MB rather than the complete audio archives. The first published sample’s WAV is 80,701,946 bytes, 457.478 seconds, stereo 44.1 kHz. No LUCID MOSS inference has run.

**The pipeline has now been compared with actual published values for 180 metadata-matched rows across 15 shared metrics.** All 60 vocoder rows have two candidate recordings and remain explicitly unavailable. Custom mean within-speaker recording APE, averaged across speakers:

| Condition | Rows | Token count | TTR | MATTR50 | HD-D42 | Original MTLD |
|---|---:|---:|---:|---:|---:|---:|
| noBarrier | 60 | 22.84% | 8.65% | 3.03% | 2.40% | 14.67% |
| babble | 60 | 9.78% | 3.87% | 2.29% | 1.37% | 7.51% |
| L2 | 60 | 9.63% | 4.24% | 1.66% | 0.96% | 7.21% |

**This is not successful exact reproduction:** neither custom nor current TAALED reproduces any scored row’s token count exactly. The CSV reproduces all four article Table 1 token means within 0.01 word, but original cleaned indexN.txt files, exact source/tier mapping and historical GUI settings are absent from the public supplement. Large no-barrier count discrepancies need source review as well as preprocessing/version checks. No alternative recording was selected by score, and no normalizer or model was tuned. See [the full LUCID report](../lucid_validation/REPORT.md), its retained pair-level results, and `outputs/validation/lucid_validation/source_audit.json`. Word-level TextGrid intervals do not supply CHAT utterance gold.

The [OCSC paper](https://talkbank.org/childes/access/Eng-NA/0docs/OCSC.pdf) provides human transcripts/audio and published MLU analyses, but its age/robot regressions have not been reproduced. The [NYU–Emerson release](https://talkbank.org/asd/access/English/NYU-Emerson.html) provides clinical reference transcripts/video, rather than competitor scores for our split. [IDEAS](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1322665/full) has published child metric errors but restricts its analyzed data to IRB-approved study members. None supplies permission to rank unrelated dataset scores against one another.

## What remains necessary for the paper

1. Resolve LUCID’s historical cleaned inputs, source/tier mapping (including vocoder ambiguity) and tool configuration, then report a separately versioned reproduction. Preserve this completed discrepancy audit. An unsent author-query draft records the exact missing information. Scope this healthy-adult corpus to lexical measurement validation.
2. Freeze a multi-participant public audio manifest, reviewed target mappings, native CHAT conventions, endpoints and compute budget before further audio scoring. Include all eligible records and retained failures; compare an established audio system on those same recordings.
3. Improve segmentation only on public development data, with independent confirmation afterward. Current exact utterance-span F1 is only **65.14% OCSC / 52.18% NYU**, despite stronger gap F1. Professional-level labeling has not been established.
4. Validate morphology, grammaticality, mazes and production-error annotation before reporting MLUm, PGU, error counts or maze percentage. Reviewed timing is needed for the defined WPM measure. Missing values stay unavailable.

The current implementation passes **78 regression tests**. Registered protocols, software/model/audio hashes, raw CLAN outputs and reproduction instructions are retained. No training job or automatic queue is running.
