# Same dataset, same language measure, published measurement error

Review date: 20 October 2026.

**Result: zero qualifying published measurement-error benchmarks verified. No defensible published-method-versus-our-method error-rate ranking can be calculated from the sources located.** This is a search finding, not proof that no such publication exists.

## Eligibility

A qualifying result must use OCSC, NYU–Emerson, or the adult Southern British English LUCID corpus, and report estimation error against a reference for at least one language measure below. Audio, transcript, manual and automatic methods are all eligible. No requirement that a competitor use TAALED or CLAN was imposed.

Measures searched: word/token and type counts (including content/function counts), lexical density, TTR and log/root/Maas variants, MATTR, MSTTR, HD-D, MTLD, utterance count, MLU in words or morphemes, utterance-length SD, words per minute, percentage grammatical utterances, production omission/substitution counts and maze percentage. Matching a metric family is only a candidate match: variants and scoring definitions would still need checking before numerical comparison.

WER/CER, speaker identification, diagnosis accuracy, speech intelligibility, phoneme/word-boundary F1, acoustic measurement error and regression model fit are not errors in these language measures. An SD or standard error describing a sample or a regression is not a method's measurement error. Speech-production errors/mazes are distinct from ASR edits.

## Closest papers: exact corpus and relevant measures, but no published measurement error

| Paper and source | Dataset | Relevant reported measures | Published error rate for these measures | Comparison with ours |
|---|---|---|---|---|
| Wagner et al. (2025), *The Ohio Child Speech Corpus*, Speech Communication 170:103206. [Full paper](https://talkbank.org/childes/access/Eng-NA/0docs/OCSC.pdf), [DOI](https://doi.org/10.1016/j.specom.2025.103206) | OCSC | Number of utterances and MLU; section 3.2 and Figures 3–4 | None found. These are descriptive/developmental and robot-condition analyses, not errors against separately established reference measurements. | Cannot rank. The paper's unspecified MLU unit must also not silently be relabeled MLUw. |
| Baese-Berk, Drake, Foster & Lee (2021), *Lexical Diversity, Lexical Sophistication, and Predictability for Speech in Multiple Listening Conditions*, Frontiers in Psychology 12:661415. [Full paper](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.661415/full) | Adult LUCID | Word tokens, TTR, MATTR50, HD-D42 and MTLD; additional TAALED measures in author data | None found. The paper tests listening-condition differences, not measurement accuracy. | Cannot rank error rates. Author values can serve as a reproducibility reference, but are not the authors' estimation errors. |
| Baker & Hazan (2011), *DiapixUK: task materials for the elicitation of multiple spontaneous speech dialogs*, Behavior Research Methods 43:761–770. [Article and supplement listing](https://link.springer.com/article/10.3758/s13428-011-0075-y), [author manuscript](https://discovery.ucl.ac.uk/1224322/1/Hazan_baker%20and%20Hazan%20BRM_revised_final.pdf) | Adult LUCID | Words produced; results subsection A; participant word counts in supplementary table | None found. Mean words produced is a sample measurement, not word-count estimation error. | Cannot rank. |
| Hazan, Uther & Granlund (2015), *How does foreigner-directed speech differ from other forms of listener-directed clear speaking styles?*, ICPhS. [Full paper](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2015/Papers/ICPHS0346.pdf) | Adult LUCID | TotalW, total words produced by Talker A; section 3.1 | None found. Means and SDs describe language production under listening conditions. | Cannot rank. |

### NYU–Emerson

The [official corpus page](https://talkbank.org/asd/access/English/NYU-Emerson.html) cites Steen et al.'s March 2026 MoLA poster, *Introducing the NYU-Emerson Corpus: Transcripts of parent–child interaction with young autistic children*. The [conference program](https://molautism.org/wp-content/uploads/sites/20/2026/03/MoLA-2026-Poster-Program.pdf) confirms the presentation; the poster itself was not retrieved. It cannot be treated as a fully inspected publication with no errors reported.

The [official transcription protocol](https://talkbank.org/asd/access/0docs/NYU-details.pdf) describes a 90% utterance-boundary/delimiter reliability **training requirement on at least two training transcripts**. This is not a published corpus-level MLU/count/lexical measurement-error result, and cannot be converted to a 10% error comparator for our metrics. No qualifying NYU–Emerson result was verified.

## Other leads and why they fail the requested comparison

| Candidate | Evidence checked | Decision |
|---|---|---|
| Shankar et al. (2025), *CHSER: A Dataset and Case Study on Generative Speech Error Correction for Child ASR*. [Official paper](https://www.isca-archive.org/interspeech_2025/balajishankar25_interspeech.pdf) | Full PDF retained; OCSC evaluation included | ASR errors, not errors estimating MLU, word count, lexical diversity or the other requested measures. |
| Hamoudi et al. (2025), *Psychiatric Voice Biomarkers: Methodological flaws in pediatric populations*. [Primary preprint](https://www.medrxiv.org/content/10.1101/2025.10.13.25337901v1.full) | Primary abstract/method/result text; PDF retrieval failed | OCSC, but WER/CER, SMD and BERTScore. No qualifying language-measure error verified. Full-PDF access limitation remains explicit. |
| *Age-Aware Adapter Tuning for Children's Speech Recognition* (2026). [Primary text](https://arxiv.org/html/2606.05440v1) | HTML methods/results and OCSC references | ASR WER, not requested language-measure errors. |
| *ChildVox* (2026). [Primary text](https://arxiv.org/html/2605.29257v1) | HTML; OCSC mention occurs in future-extension discussion | OCSC citation does not establish OCSC evaluation. Its language-level evaluation uses other data. |
| Gao & Sun (2025), *A Computational Simulation of Language Production in First Language Acquisition*. [Official paper](https://aclanthology.org/2025.emnlp-main.557.pdf) | Primary paper; OCSC listed among numerous CHILDES corpora | No separate OCSC estimation-error result for our language measures verified. Generation scores cannot substitute. |
| Shen et al. (2026), *Evaluating Cognitive Age Alignment in Interactive AI Agents*. [Primary text](https://arxiv.org/html/2605.17894v1) | HTML, including calibration and feature appendices | Corpus-calibrated distributions and generated-agent complexity are not OCSC MLU/MATTR estimation errors. |
| Brahimi et al. (2026), VAP turn-taking study. [Official paper](https://aclanthology.org/2026.iwsds-1.34.pdf) | Primary paper screened during this literature search | SHIFT/HOLD prediction is not MLU, utterance-count or lexical measurement error. |
| DrivenData, *pasketti-word*. [Official model card](https://huggingface.co/drivendata/pasketti-word) | Evaluation description and corpus table | OCSC included, but WER benchmark; not a qualifying language-measure error paper. |
| Ludusan, Synnaeve & Dupoux (2015), *Prosodic boundary information helps unsupervised word segmentation*, NAACL-HLT, pp. 953–963 | Full PDF retained | Adult LUCID is used, but word-segmentation/prosodic-boundary scores are not errors in utterance count, MLU or word-count estimates. |
| Kruyt et al. (2023), *Measuring Prosodic Entrainment in Conversation: A Review and Comparison of Different Methods*. [Primary paper](https://doi.org/10.1044/2023_JSLHR-23-00094) | Full PDF retained | Adult LUCID, but prosodic entrainment measurements rather than the requested language measures. |
| Hazan & Baker (2011), *Acoustic-phonetic characteristics of speech produced with communicative intent to counter adverse listening conditions*. [DOI](https://doi.org/10.1121/1.3623753) | Full PDF retained | Adult LUCID, but acoustic/communication outcomes; no qualifying language-measure estimation error found. |
| Barrett et al. (2025), *Autoscribe: An automated tool for creating transcribed TextGrids from audio-recorded conversations*. [Full paper](https://link.springer.com/article/10.3758/s13428-025-02850-9) | Full publisher HTML, corpus definitions and accuracy methods/results | UCL corpus contains children/adolescents aged 9–14, not adult LUCID; other corpus is US adults. Boundary timing and transcription accuracy also differ from target metric errors. |
| Russell et al. (2024), *What automatic speech recognition can and cannot do for conversational speech transcription*. [Publisher record](https://www.sciencedirect.com/science/article/pii/S2772766124000697) | Abstract and metadata; publisher full text blocked | No qualifying exact-corpus language-measure error verified. Not asserted to have been fully reviewed. |
| *Studying Conversational Adjustments in Interaction: Beyond Acoustic Phonetic Changes* (2023). [DOI](https://doi.org/10.1044/2023_JSLHR-23-00268) | Abstract | Wildcat corpus, not our LUCID recordings. Shared Diapix task does not make it the same dataset. |
| *Noise-induced differences in the complexity of spoken language* (2022). [DOI](https://doi.org/10.1177/17470218221124869) | Abstract | Newly elicited picture-description conversations via videoconferencing; a relevant citation does not establish our exact dataset. |

Other common false matches excluded: LuCiD Language 0–5, kidLUCID/elderLUCID, new recordings elicited with Diapix pictures, and unrelated corpora named LUCID.

## Search coverage and reproducibility

- Exact corpus-name OpenAlex searches returned 41 records for adult-LUCID's name, six for Ohio Child Speech Corpus and one for NYU–Emerson. These are search hits, not counts of eligible studies.
- Forward citation searches retrieved 183 records citing Hazan & Baker's acoustic study, all 202 records citing Baker & Hazan's DiapixUK study (200 + two on the next page), and 19 citing Baese-Berk et al. The overlap is retained and deduplicated in `index_screening.json`.
- Across those seven index responses, 400 unique records were retained. Screening was by titles and available abstracts, with primary-paper follow-up for promising candidates. **This does not mean 400 full papers were read.** Nine retrieved full PDFs and their extracted texts are retained in `sources/`; additional primary HTML was inspected online.
- Public web searches combined exact dataset names and spelling variants with MLU, MLUw, utterance count/SD, word/token/type count, TTR, MATTR, MSTTR, HD-D, MTLD, lexical density, grammatical utterances, mazes, omission/substitution, error, accuracy, agreement and absolute/relative error. References, publisher supplementary-material listings and the TAALED paper's author data were checked where accessible.
- The Baese-Berk author TAALED CSV was independently retrieved earlier in this same research session and byte-matched to our retained reference. Its provenance receipt and CSV remain in `../published_metric_values_2026-10-20/`. These are published measurements, not the paper's own accuracy estimates. The paper's plotted MTLD variant and our original MTLD are not interchangeable.
- Baker & Hazan's publisher page labels its supplementary DOCX as participant word counts averaged across three tasks. The DOCX retrieval through the browser failed; the table contents were not claimed inspected. The manuscript's description and publisher listing do not report a measurement-error comparison.
- Limitations include incomplete indexing, abstracts without full text, an unretrieved NYU poster, an unavailable supplement and failed publisher retrievals. Absence from this review must not be described as proof of absence from all literature.

## Comparison decision

| Dataset | Verified published error rates for requested language measures | Better/worse than our pipeline |
|---|---|---|
| OCSC | None located | Undetermined |
| Adult LUCID | None located | Undetermined |
| NYU–Emerson | None located; poster full text unavailable | Undetermined |

No missing external error was filled with zero, our own error, a descriptive mean, a regression coefficient, WER, F1 for another task, or an error from a different corpus. No experiments or rescoring were run for this review.

### Correction to earlier comparisons

Differences between our aggregate word counts and the Baker/Hazan papers' aggregate word counts are **not published competitor error rates**. Nonmatched sample/aggregation differences cannot determine which method is closer to ground truth. Earlier such percentage differences must not be used in a method-accuracy comparison or superiority claim. Existing files are preserved as history; this report provides the stricter interpretation requested here.
