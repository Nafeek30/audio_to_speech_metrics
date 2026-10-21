# Exact datasets → published metric results

20 October 2026. Scope follows the user's clarification: report published values of our metrics on the same datasets, regardless of whether the original input was audio or transcription. This document contains external results, not our model's scores.

## 1. Baese-Berk et al. (2021) — LUCID — TAALED

*Lexical Diversity, Lexical Sophistication, and Predictability for Speech in Multiple Listening Conditions*. Frontiers in Psychology 12:661415. [Article](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.661415/full), DOI 10.3389/fpsyg.2021.661415. The paper links its [author-released data](https://osf.io/dfhpu/?view_only=49d95d90424941da82217a239ab7450c) in Footnote 2 and the data-availability statement.

**Condition means calculated directly from the authors' published TAALED CSV** (60 sample rows per condition; 20 speakers per condition):

| Metric | No barrier | Babble | L2 | Vocoder |
|---|---:|---:|---:|---:|
| Word tokens | 662.78 | 756.75 | 1095.92 | 785.22 |
| Word types | 205.75 | 184.42 | 230.75 | 206.73 |
| TTR, all words | 0.3174 | 0.2534 | 0.2157 | 0.2737 |
| MATTR50, all words | 0.6975 | 0.6242 | 0.6508 | 0.6534 |
| MSTTR50, all words | 0.6972 | 0.6243 | 0.6505 | 0.6551 |
| HD-D42, all words | 0.7690 | 0.7296 | 0.7391 | 0.7465 |
| Original MTLD, all words | 39.80 | 28.71 | 32.03 | 32.10 |

These arithmetic summaries are derived from published per-sample measurements; they are not all printed as a numerical table in the article. Article Table 1 prints token means of 662.78, 756.75, 1095.92 and **785.21**, respectively. The vocoder discrepancy of 0.01 remains explicit. Figures 1–3 plot TTR, MATTR50 and HD-D42; Figure 4 uses a different moving-average MTLD variant, not original MTLD.

The remote author CSV was independently retrieved successfully on 20 October 2026: HTTP 200, 94,436 bytes, 240 rows, SHA256 `f4e9ad97e50d859311f04502a2c462826a783b064a5e4ce6e4282b058b947b63`, identical to the retained reference. Source copy, transfer headers and verification receipt are saved here. All 15 shared metric means and sample SDs are in `lucid_published_condition_means.csv`.

## 2. Baker & Hazan (2011) — LUCID — word count

*DiapixUK: a task for the elicitation of spontaneous speech dialogs*. Behavior Research Methods 43, 761–770. [Author manuscript at UCL](https://discovery.ucl.ac.uk/1224322/1/Hazan_baker%20and%20Hazan%20BRM_revised_final.pdf).

The study uses LUCID recordings from 40 native Southern British English speakers, pairs of friends completing three successive picture tasks in good listening conditions.

**Published mean number of words per participant per conversation: 613.** Location: results subsection A, “Does the task provide enough speech material for acoustic/linguistic analysis?”, manuscript p. 13. This is their word-count definition and sample aggregation; it is not an independently reported TAALED result.

## 3. Hazan, Uther & Granlund (2015) — LUCID — total words

*How does foreigner-directed speech differ from other forms of listener-directed clear speaking styles?* Proceedings of ICPhS 2015. [Official full paper](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2015/Papers/ICPHS0346.pdf).

Published **TotalW**, words produced by Talker A, means and SDs from section 3.1, PDF p. 3:

| Speaker group | Condition | Mean total words | SD |
|---|---|---:|---:|
| VOC/L2, 20 talkers | No barrier | 1825 | 390 |
| VOC/L2, 20 talkers | Vocoder | 2365 | 579 |
| VOC/L2, 20 talkers | L2 | 3170 | 699 |
| VOC/BAB, 20 talkers | No barrier | 1851 | 535 |
| VOC/BAB, 20 talkers | Vocoder | 2191 | 826 |
| VOC/BAB, 20 talkers | Babble | 2157 | 521 |

Three picture tasks were completed in each condition. These are totals at that broader aggregation, not the per-conversation counts in the 2021 TAALED paper. The two rows for the same condition concern different 20-speaker groups.

## 4. Wagner et al. (2025) — OCSC — MLU and utterance count

*The Ohio Child Speech Corpus*. Speech Communication 170:103206. DOI 10.1016/j.specom.2025.103206. [Official full paper](https://talkbank.org/childes/access/Eng-NA/0docs/OCSC.pdf).

Section 3 analyzes **301 children performing Narrative Pictures**. Exact published numerical results on our metric families, section 3.2, printed p. 6:

| Outcome | Reported result |
|---|---|
| MLU | Age coefficient β=0.321, p<0.001; regression R²=0.112 |
| MLU | Robot-role coefficient β=0.071, p=0.198 |
| Utterance count | Age coefficient β=8.169, p=0.019; regression R²=0.074 |
| Utterance count | Robot Absent vs Robot Encouragement contrast β=82.335, p=0.043 |

These are regression coefficients/results on MLU and count, **not raw mean MLU/count values or automatic measurement errors**. Individual MLU and count results are plotted in Figures 4 and 3. No exact numerical table of raw group means was verified, and none is reconstructed or invented from the graph. The paper's MLU must not silently be relabeled MLUw or MLUm without a verified definition.

## NYU–Emerson

No paper with numerical values for MLU, utterance count, NDW, TTR, MATTR, HD-D or MTLD on the released NYU–Emerson corpus was verified. The [official release documentation](https://talkbank.org/asd/access/English/NYU-Emerson.html) and 2026 introduction-poster citation do not provide such a table. No value is supplied in place of a missing result.

## Extraction and search limits

The first three papers provide exact word-count or lexical-measure values; the fourth provides exact statistical results for MLU/count and graphical raw observations. This distinction is retained rather than calling all four papers raw-value tables. The LUCID metrics are author-profile values; no claim is made that every original paper used CLAN or TAALED, or that their different sample aggregations can be mixed.

Additional OpenAlex searches and citation checks identified OCSC ASR papers, an acquisition simulation pooling many CHILDES corpora, and an age-alignment paper using OCSC among other sources. None supplied a verified separate OCSC raw table of our metrics. The corpus source's citation index missed known directly verified papers, so index counts are not evidence that a literature search is exhaustive. Other LUCID studies reporting acoustic, recognition or prosodic segmentation outcomes were not substituted for the requested metrics. No new inference or scoring was run.
