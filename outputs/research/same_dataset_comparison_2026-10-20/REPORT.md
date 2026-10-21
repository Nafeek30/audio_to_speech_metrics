# Same-dataset literature and saved-result comparison

Research date: 20 October 2026. Scope: OCSC, NYU–Emerson, and the adult Southern British English LUCID corpus used by Baese-Berk et al. No training, inference, scoring, model selection, or audio acquisition was run for this review. All project errors below are copied from existing results, not newly evaluated estimates.

## Answer

The search did **not identify three or four published, directly comparable TAALED/CLAN measurement-error benchmarks on these datasets**. Four genuine papers using these datasets are listed below, but they do not form a leaderboard for our task. One supplies direct TAALED reference values; another analyzes human-transcribed utterance count and MLU; the others address different prediction or acoustic outcomes. NYU–Emerson's verified release citation is a conference poster, not a journal benchmark.

We cannot conclude that the current pipeline outperforms three or four published systems on these datasets. We can report agreement with reference measurements, the remaining errors, and a separate, qualified earlier comparison against an official Batchalign segmentation component.

## Four verified papers using the same corpora

| Paper | Corpus and published endpoint | Verified numerical result or source location | Valid comparison with us |
|---|---|---|---|
| Baese-Berk, M. M., Drake, S., Foster, K., Lee, D., Staggs, C., & Wright, J. M. (2021). *Lexical Diversity, Lexical Sophistication, and Predictability for Speech in Multiple Listening Conditions*. Frontiers in Psychology 12:661415. [Article](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.661415/full), DOI 10.3389/fpsyg.2021.661415. | Adult LUCID; TAALED lexical statistics, plus TAALES/predictability analyses. | Table 1 average tokens: no-barrier 662.78; L2 1,095.92; babble 756.75; vocoded 785.21. Figures 1–3 show TTR, MATTR50 and HD-D42. Footnote 2 links author-exported results/code on OSF. | Direct lexical **reproduction/agreement** target. The paper does not report a competing pipeline's percentage errors against human references. Higher lexical values do not mean a better measuring system. |
| Wagner, L., et al. (2025). *The Ohio Child Speech Corpus*. Speech Communication 170:103206. [Publisher](https://www.sciencedirect.com/science/article/pii/S0167639325000214), [full paper](https://talkbank.org/childes/access/Eng-NA/0docs/OCSC.pdf), DOI 10.1016/j.specom.2025.103206. | OCSC; professionally reviewed CHAT transcripts, utterance count and MLU in the Narrative Pictures task for 301 children. | Section 3.2, printed p. 6: utterance-count regression R²=0.074; MLU regression R²=0.112. Figures 3–4 describe utterance count and MLU by age/robot condition. | Human reference measurements and scientific analyses, **not automated measurement accuracy**. R² here is about age/robot predictors, not agreement of automatic and human MLU. Their task/population aggregation differs from our 45-child, multi-task test subset. No numerical superiority comparison is justified. |
| Brahimi, Y., Blanc, C., & Fourtassi, A. (2026). *Predicting Turn-Taking in Child–Adult Conversations Using Voice Activity Projection*. Proceedings of IWSDS, pp. 338–347. [Publication](https://aclanthology.org/2026.iwsds-1.34/), [paper](https://aclanthology.org/2026.iwsds-1.34.pdf). | OCSC and Switchboard; predicting whether speech after a silence is a speaker SHIFT or HOLD, using audio. | Study 1 / Table 2, printed p. 343: OCSC original VAP F1=90.28%, balanced accuracy=93.96%. | Genuine same-corpus prediction results, but **different labels, input, sample construction and test split**. These scores cannot be ranked against our word-gap boundary F1 or CLAN/TAALED errors. |
| Guo, Z.-c., & Smiljanic, R. (2023). *Speakers coarticulate less in response to both real and imagined communicative challenges: An acoustic analysis of the LUCID corpus*. Journal of Phonetics 97:101210. [Publisher](https://www.sciencedirect.com/science/article/pii/S0095447022000857), DOI 10.1016/j.wocn.2022.101210. | Same adult LUCID corpus; spectral distance and coarticulatory overlap in interactive and read speech. | Publisher abstract identifies acoustic measurements and communicative-condition effects. No TAALED/CLAN measurement-error comparator was verified. Full publisher text was access-limited. | Related same-corpus research; excluded from numerical accuracy comparison. Included to distinguish a real same-dataset paper from a usable benchmark. |

The Guo & Smiljanic 2021 Interspeech precursor, *Speakers coarticulate less when facing real and imagined communicative difficulties: An analysis of read and spontaneous speech from the LUCID corpus*, is also real ([official paper](https://www.isca-archive.org/interspeech_2021/guo21b_interspeech.pdf)); it is not counted as an independent TAALED/CLAN comparator.

### NYU–Emerson

The [official corpus page](https://talkbank.org/asd/access/English/NYU-Emerson.html) cites Steen, K., Buonocore, T., Luyster, R., Sancimino, C., & Arunachalam, S. (2026, March), *Introducing the NYU-Emerson Corpus: Transcripts of parent–child interaction with young autistic children*, a Meeting on Language in Autism poster. The [official poster program](https://molautism.org/wp-content/uploads/sites/20/2026/03/MoLA-2026-Poster-Program.pdf) confirms poster 55. The full poster was not retrieved; no numerical metric-error result is inferred from its title.

The current release contains 30 children and is a subset of a larger study. A paper from the same laboratories or broader study is not automatically a study of these exact released recordings. No comparable automated TAALED/CLAN error estimate for this release was verified in this search.

The [transcription procedure](https://talkbank.org/asd/access/0docs/NYU-details.pdf) mentions reaching 90% reliability for utterance boundaries/delimiters on at least two training transcripts as a human annotator training criterion. This is **not** a published corpus-wide model accuracy estimate and cannot be compared directly with our boundary F1 or percentage error in MLU. Its mention of Batchalign2 proofreading also supplies no Batchalign numerical accuracy result on the release.

### Other screened work

*Psychiatric Voice Biomarkers: Methodological flaws in pediatric populations*, medRxiv preprint DOI [10.1101/2025.10.13.25337901](https://www.medrxiv.org/content/10.1101/2025.10.13.25337901v1.full), describes OCSC ASR evaluation using WER/CER and other transcript metrics. Indexed primary-source text identifies gold utterance timestamps and speaker-role information. Direct full-page access returned 403, so exact numerical tables/authorship were not independently extracted for this report. It is not used as a TAALED/CLAN comparator, and ASR error is not interchangeable with downstream metric error.

The Pasketti children's ASR resource also includes OCSC, but an ASR resource/training-data listing does not establish a same-split CLAN/TAALED benchmark. ENNI SATE results from the earlier general literature memo use another dataset and are excluded here. The child-language LuCiD/Language 0–5 project is not this adult LUCID corpus. A preprint and its final paper are not two independent comparators.

## Actual saved LUCID agreement with the author reference data

Source: `outputs/results/transcript_validation/lucid_summary.csv`. Mean absolute percentage error (APE), calculated within sample, then within speaker, then across speakers. Lower is closer to the reference. These are **our** errors against author values, not errors reported by the paper.

| Metric | Babble APE % | L2 APE % | No-barrier APE % |
|---|---:|---:|---:|
| Word tokens | 0.76 | 1.87 | 16.60 |
| Word types | 1.76 | 1.33 | 12.05 |
| TTR | 1.31 | 1.57 | 7.56 |
| MATTR50 | 0.44 | 0.44 | 2.11 |
| HD-D42 | 0.37 | 0.25 | 2.12 |
| Original MTLD | 2.49 | 2.42 | 10.49 |

Coverage is 60 samples / 20 speakers per scored condition, 180 of the 240 author-export rows in total. The 60 vocoder rows remain unresolved and are not silently excluded from the coverage account. All 15 shared exported metrics satisfy our declared engineering targets in babble/L2; several no-barrier measures do not. Those thresholds are project goals, not published clinical acceptance standards.

**Publication provenance remains provisional.** The local author-results CSV agrees with retained acquisition metadata, but an independent same-version remote byte confirmation remains pending. The article itself and its OSF data link were verified; this does not independently authenticate our retained CSV. See `outputs/results/transcript_validation/LUCID_METRIC_PROVENANCE.md`. Original MTLD is an author-export column and differs from the moving-average bidirectional MTLD in the article's Figure 4. Do not describe all 15 exported metrics as 15 main-article outcomes. The local vocoder token mean is 785.22 versus article Table 1's 785.21; the 0.01 difference is unresolved.

Interpretation: close lexical agreement in babble/L2, incomplete overall reproduction, no claim to outperform TAALED. TAALED supplies the reference calculations we are trying to reproduce.

## Current production results against human CHAT / native CLAN

Source: `outputs/results/transcript_validation/REPORT.md`, bounded-calibration results, and `chat_calibrated_results.json`. Retained observed-turn MiniLM workflow, margin 0.25. Existing test participants have historical exposure; this is not newly unexposed external confirmation. OCSC: 45 children / 111 sections; NYU: 4 children / 15 sections. Errors below use mean participant APE.

| Metric | OCSC APE % | NYU–Emerson APE % |
|---|---:|---:|
| Total utterances | 10.79 | 26.45 |
| Eligible MLU utterances | 10.36 | 29.98 |
| MLU in words (MLUw) | 8.70 | 16.57 |
| Utterance-length SD | 14.24 | 13.70 |
| Total words (TNW) | 1.86 | 2.34 |
| Different words (NDW) | 1.79 | 1.69 |
| TTR | 1.14 | 1.90 |
| MATTR | 0.49 | 1.97 |

The reference path uses native CLAN with original human CHAT coding. Lexical counts/ratios are closer than utterance-dependent metrics. Count/MLUw/SD remain above our engineering goals. NYU's four-child test is too small for broad generalization. An MLUw APE of 8.70% is not 91.30% classification accuracy, and does not say that each recording is within 8.70%. We have not reproduced the OCSC source paper's exact Narrative Pictures aggregation, so raw population means are not ranked here.

## Separate existing competitor comparison: a narrow better-than claim

`outputs/results/common_benchmark/REPORT.md` contains an earlier matched-input comparison on unpunctuated target-child paragraphs using normalized spoken-word references. It compared an earlier `latest_minilm` paragraph workflow with official Batchalign 0.10.4 CHATUtteranceBackend, not the current calibrated observed-turn workflow and not the full published Batchalign audio system.

| Corpus | Method | Count APE % | MLUw APE % | Length-SD APE % |
|---|---|---:|---:|---:|
| OCSC | Earlier paragraph MiniLM | 10.79 | 8.75 | 16.20 |
| OCSC | Batchalign 0.10.4 segmentation component | 52.47 | 113.64 | 121.48 |
| NYU–Emerson | Earlier paragraph MiniLM | 24.91 | 16.99 | 18.87 |
| NYU–Emerson | Batchalign 0.10.4 segmentation component | 56.48 | 106.04 | 77.40 |

This supports: **our earlier adapted paragraph workflow had lower count/MLUw/SD errors than the unadapted Batchalign component under that recorded protocol**. The models did not receive equal domain adaptation. Historical test exposure, four NYU test children, the different reference profile, and component-only scope must accompany the claim. These numbers are our evaluation of external software; they are not the Batchalign paper's reported OCSC/NYU results. Do not insert the current production errors into this old comparison.

## What can be claimed in a manuscript

“We evaluated agreement with author-released TAALED measurements on adult LUCID and with professional CHAT-derived native CLAN measurements on participant-disjoint OCSC and NYU–Emerson cohorts. Agreement varied across metrics and conditions; utterance-dependent errors and unresolved reference coverage remain limitations. An earlier controlled paragraph benchmark showed lower downstream errors than an official pretrained Batchalign segmentation component under the specified protocol.”

Before publication, resolve the LUCID remote provenance qualification; report the exposed-test history and small NYU cohort. To support a stronger current-workflow superiority claim, competitors must eventually be evaluated under matching inputs, reference policies and aggregation, with appropriate uncertainty. That is future evaluation work; it was not run during this paused literature review.

## Search scope and verification limits

Searched exact corpus names and combinations with TAALED, CLAN, MLU, lexical diversity, accuracy, error, ASR and utterance segmentation; checked TalkBank source documentation/citations, publisher pages, Frontiers/PMC, ACL Anthology, ISCA, medRxiv and the NYU release conference program. Representative exact queries included:

- `"Ohio Child Speech Corpus" "MLU"`
- `"Ohio Child Speech Corpus" "lexical" diversity`
- `"Ohio Child Speech Corpus" "TAALED" accuracy`
- `"NYU-Emerson" corpus MLU TAALED CLAN`
- `"NYU-Emerson" "MLU"`
- `"NYU Emerson" corpus CLAN accuracy`
- `"LUCID" "TAALED" lexical diversity`
- `"LUCID corpus" "TAALED" error accuracy`

This is a targeted literature search, not proof that no unindexed, inaccessible or unpublished relevant result exists. Unsupported comparisons and inaccessible numerical tables were not filled in. No researcher contact occurred.
