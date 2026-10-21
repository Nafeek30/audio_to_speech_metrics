# Published benchmarks and the evidence available for this pipeline

Research checked 20 October 2026. This is documentation and literature review only. No training, inference, scoring, selection, audio acquisition, or production change was performed. Earlier results and protocols remain unchanged.

## What the meeting feedback means

Published numerical comparators exist. They help establish the research context and evaluation methods. A claim of superior performance is strongest when competing systems process the same recordings or transcripts under the same scoring rules. Lower numbers from unrelated papers do not by themselves establish superiority.

There is also an existing matched-input competitor experiment in this project. It should be presented alongside the literature, with its historical model versions and limits stated. A practical, domain-adapted system comparison can support a narrower claim than an architecture comparison or a claim of universal superiority.

## The most useful primary papers

### 1. Pelfrey et al. (2024): IDEAS

[Evaluating the accuracy of automated processing of child and adult language production in preschool classrooms](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1322665/full). DOI: 10.3389/fpsyg.2024.1322665.

Table 4, children's own talk, reports median absolute relative errors: utterance count **11.11%**, word-based MLU **9.29%**, unique words **29.17%**, words **30.00%**, and TTR **17.43%**. Six-minute segments and manually supplied timestamps were used. These are useful downstream-error benchmarks, but include automatic transcription error and use different aggregation and reference conventions from our participant-averaged CHAT results. Do not treat them as clinical acceptance thresholds or a head-to-head comparison.

### 2. Hou et al. (2026): SATE

[Speech Annotation and Transcription Enhancer (SATE): An Automated System for Child Language Sample Analysis](https://doi.org/10.1145/3832036). Published 30 September 2026. The [author-hosted full paper](https://cse.buffalo.edu/~wenyaoxu/papers/journal/xu-imwut2026.pdf) was downloaded, its 37 pages extracted, and Tables 5 and 9 visually verified. The source copy and table-page images are in `sources/` beside this note.

The study uses ENNI narratives, with child-level separation of training and test data. Table 5 evaluates segmentation on ground-truth text: SATE precision/recall/F1 **93.9/94.3/94.1%**, released Batchalign **84.3/93.0/88.4%**, and SaT 3L **74.2/47.7/58.1%**. Only the authors' segmentation model received ENNI-specific fine-tuning; the paper explicitly frames this as a practical deployment comparison.

Table 9 reports these complete-system **Pipeline Solution** results against expert-derived measures:

| Measure | Pearson r | Mean absolute error in original units | Mean signed difference | Mean percentage difference reported |
|---|---:|---:|---:|---:|
| Total utterances, NTU | .92 | 19.88 utterances | +18.03 utterances | +19.64% |
| Total words, NTW | .98 | 32.78 words | +16.89 words | +2.24% |
| Different words, NDW | .97 | 9.83 word types | +7.48 types | +5.03% |
| MLU in morphemes, MLUm | .78 | 1.89 morphemes/utterance | -1.85 morphemes/utterance | -25.69% |

Keep this as one consistent system row. Other variants win some individual measures, but do not assemble their minima into one fictitious system. Signed differences are not absolute percentage errors; the paper's SD of differences is not our within-sample utterance-length SD. MLUm cannot be ranked against MLUw. This is an important current competitor and demonstrates why correlations alone are inadequate. The paper also reports 11.81% ASR WER; that is a separate endpoint from metric accuracy.

### 3. Liu et al. (2023): Batchalign

[Automation of Language Sample Analysis](https://talkbank.org/aphasia/publications/2023/Liu23.pdf). DOI: 10.1044/2023_JSLHR-22-00642.

Table 1 reports utterance-tokenization F1 **86.9%** for adult MCI speech and **85.0%** for controls; the prose instead says **85.1%** for controls. Preserve this discrepancy. This is a close CHAT pipeline precedent, but adult clinical speech and different input/scoring conventions prevent ranking it against our child-corpus results. Its published full pipeline is distinct from a local evaluation of its released segmentation component.

### 4. Fox et al. (2021): school-age DLD narrative transcription

[An Evaluation of Expedited Transcription Methods for School-Age Children's Narrative Language: Automatic Speech Recognition and Real-Time Transcription](https://pubs.asha.org/doi/10.1044/2021_JSLHR-21-00096).

Forty-two narratives were evaluated. Table 5 reports ASR-versus-reference correlations: utterance count **.97**, MLU-words **.89**, NDW **.95**, TNW **.94**. Mean weighted WER was **30%**. These are useful precedents for validating downstream language measures despite imperfect transcripts. Correlation is not percentage accuracy and cannot be converted into MAPE.

### 5. Sun et al. (2025): WSW 2.0

[Who Said What (WSW 2.0)? Enhanced Automated Analysis of Preschool Classroom Speech](https://arxiv.org/html/2505.09972v1). The verified source is the arXiv manuscript; numerical locations below refer to that version.

Table V reports integrated-pipeline child absolute-agreement ICC **.713** for overall MLU and **.863** for lexical diversity, defined as unique words per minute. Child transcription WER is **23.8%**. The prose separately gives **.74** for child MLU; use the explicitly labeled table endpoint when citing the full integrated system. This is audio-based preschool evaluation, not a MATTR benchmark or a percentage-error result.

### 6. Hayes & Kan (2026): AI calculations versus SALT

[Convergence between AI chatbot-calculated microstructure measures and SALT analysis of child language samples](https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2026.1929126/full). DOI: 10.3389/flang.2026.1929126. Published 7 September 2026.

On 84 children's coded narrative samples, Table 3 gives absolute-agreement ICCs for ChatGPT/Copilot/MagicSchool/Gemini: MLUw **.93/.92/.97/.81**, total words **.94/.94/.97/.76**, and NDW **.74/.73/.97/.81**. This is a transcript/calculator comparison under SALT conventions. The ICCs are not error percentages and do not demonstrate boundary recovery from unpunctuated text. It is a useful recent reference for reporting agreement rather than only correlation.

### 7. Ortiz, Huang & Benway (2026): newly published LSA+

[Automated Language Sample Analysis for Children With Developmental Language Disorder](https://doi.org/10.1044/2026_JSLHR-25-00543). Publication date **19 October 2026**, confirmed using the publisher-deposited Crossref record.

The verified abstract describes 66 children with DLD and 285 with typical language and reports correlations **r=.79-.98** between automated and manual measures. **Abstract-only verification:** the publisher's full tables were inaccessible in this review. Do not invent a WER, MAPE, per-measure correlation, or diagnostic accuracy from this abstract. This is an important related-work citation rather than a verified quantitative ranking.

## Coverage of our measures

| Our endpoint | Useful numerical literature | Remaining comparison issue |
|---|---|---|
| Utterance count | IDEAS percentage error; SATE MAE/bias; Fox correlation | Total units versus MLU-eligible units, context, segment length, and aggregation |
| MLUw | IDEAS percentage error; Fox correlation; WSW and Hayes/Kan ICC | Match word-based definitions and exclusions; do not substitute MLUm |
| Utterance-length SD | No directly interchangeable automated percentage-error benchmark identified in this search | SATE's SD of errors and SD of participants' mean MLU are different quantities |
| TNW and NDW | IDEAS errors; SATE MAE/bias; Fox correlation; Hayes/Kan ICC | Match fillers, retraces, unintelligibility, word types, and input stage |
| TTR | IDEAS percentage error | Match tokenization and count eligibility |
| MATTR, MTLD, HD-D and related lexical measures | Standard implementations and same-input/published-value reproduction are useful references | No comparable child-pipeline error norm verified here; match window size and algorithm variant |
| Boundary precision/recall/F1 | SATE; Batchalign | CHAT utterances versus C-units, gap masks, speaker context, and corpus |

This is a targeted literature search, not a systematic review or proof that no other papers exist. Our LUCID analysis remains lexical-only and publication-provisional pending independent same-version source reconfirmation. It is not a competing segmentation system's accuracy result.

## We already have a stronger kind of comparator: matched inputs

The preserved [common-input benchmark](../../results/common_benchmark/REPORT.md) evaluates the released **Batchalign 0.10.4 segmentation component** and earlier MiniLM snapshots on the same 126 unpunctuated task paragraphs from 49 public test children. Each method received the same target-child words and no source-turn cuts.

The following is the **earlier paragraph experiment's OCSC subset**, not current observed-turn production, not a new result, and not the stopped CV study:

| Earlier method on the same 45 OCSC children / 111 paragraphs | Count APE % | MLUw APE % | Length-SD APE % | Boundary F1 % |
|---|---:|---:|---:|---:|
| `latest_minilm` in that archived experiment | 10.79 | 8.75 | 16.20 | 85.56 |
| Released Batchalign 0.10.4 component | 52.47 | 113.64 | 121.48 | 56.31 |

These use that benchmark's normalized spoken-word references and participant-balanced error averaging. The original-coded-CHAT sensitivity audit is separately preserved in [ORIGINAL_CHAT_AUDIT.md](../../results/common_benchmark/ORIGINAL_CHAT_AUDIT.md); its numbers must not be substituted into this table.

Important limits:

- Our MiniLM was adapted on different children from these corpus domains; the external component was used without local adaptation. This is a practical comparison, not an equal-training architecture study.
- The comparison evaluates a segmentation component, not Batchalign's complete audio-to-CHAT pipeline or its original adult study.
- The test cohort has historical exposure. Report the result as descriptive benchmark evidence, not newly unexposed confirmation.
- The archived label `latest_minilm` is relative to that experiment. It does not mean a newly selected model or the retained production workflow.
- The small NYU test subset contains only four children; preserve all NYU results and limits even if a future audio study is OCSC-only.

## Current production results remain a separate table

Copied from the latest [handoff](../../../HANDOFF_2026-10-20.md) and [stopped-study report](../../results/cv_hyperparameter_v4/REPORT.md):

| Retained observed-turn production, existing test | Eligible-count APE % | MLUw APE % | Length-SD APE % |
|---|---:|---:|---:|
| OCSC | 10.36 | 8.70 | 14.24 |
| NYU-Emerson | 29.98 | 16.57 | 13.70 |

Do not compare 8.70% with IDEAS's 9.29% and conclude superiority: IDEAS reports a segment median from automatic transcription, while ours reports participant-averaged errors from supplied text with native CHAT/CLAN eligibility. Do not relabel 8.70% error as 91.30% classification accuracy. The earlier 5%/10% targets are project engineering aims, not published clinical acceptance standards. CV tuning values must remain separate from these production-test values.

## What we can say in a paper or meeting

Supported, with the methods and limits attached:

> Published automated language-sample systems report downstream errors and agreement for utterance count, MLU, word counts, and lexical measures. We evaluate these quantities directly against professional references and also report a matched transcript-input comparison with a released segmentation baseline.

For the archived direct comparison:

> In a descriptive OCSC paragraph benchmark, the domain-adapted MiniLM segmentation component produced lower downstream count, MLUw, and utterance-length-SD errors than the unadapted released Batchalign 0.10.4 segmentation component under matched word inputs and scoring rules.

Not established: overall superiority to SATE/IDEAS/full Batchalign; human-level annotation; clinical interchangeability; end-to-end audio performance across a cohort; or current MiniLM superiority from a fresh independent study.

My recommended paper framing is **measurement validation and practical baseline comparison**, with explicit failure cases and scope. A useful contribution need not win every published score. Novelty and clinical usefulness still require evidence; publication is not guaranteed by the existing results.

## Recommended evaluation additions, only if separately resumed

No step below was run in this literature task.

1. Declare transcript-input segmentation and end-to-end audio analysis as separate tasks. Use existing matched transcript results for the former; the single historical MOSS pilot is insufficient for the latter.
2. On a predeclared common input set, compare the retained full method, a MiniLM-only ablation, spaCy+rules, and a released segmentation baseline with the same available speaker context. The old three-way figure uses an earlier small transformer and is not a current MiniLM ablation.
3. Report boundary precision/recall/F1 and exact spans together with raw-unit MAE, participant-averaged APE, signed bias, agreement limits, missingness, and participant-paired uncertainty. Retain per-corpus results. Correlation/ICC and error percentages answer different questions.
4. If a clinical-acceptability claim is intended, predeclare meaningful error tolerances and evaluate human-human and model-human agreement on the same annotation task. Do not import a tolerance from unrelated ages or conventions.
5. Preserve all original cohorts, historical test-exposure disclosures, stop guards, weights, and source transcripts. A future OCSC-only MOSS study requires its separate scope/prefix-policy amendment before experiments.

## Slide-ready bullets

- Existing systems provide published benchmarks for utterance segmentation and downstream language-measure accuracy.
- IDEAS reports child-speech median errors of 11.11% for utterance count and 9.29% for MLUw; its protocol differs from ours.
- SATE is a recent child-LSA comparator with segmentation and downstream-error tables; it reports MLUm rather than MLUw.
- Our strongest comparative evidence is an existing same-input evaluation against the released Batchalign segmentation component.
- We report calculator correctness, segmentation accuracy, and downstream metric agreement separately, including limitations and failure cases.
- The intended claim is measured agreement and practical benefit on specified inputs, rather than universal superiority from cross-paper numbers.
