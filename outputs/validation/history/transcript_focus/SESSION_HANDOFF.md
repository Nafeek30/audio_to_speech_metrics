# Speech pipeline current handoff

## October 15 continuation: public benchmarks now take priority

The user's latest instruction excludes the original ten participants for now. Prioritize validated language metrics on common public data, then professional CHAT utterance segmentation. No private-cohort rescoring, new training or model selection was performed in this continuation. Historical sections below remain evidence, not the current task scope.

**New completed work:** `outputs/results/common_benchmark/REPORT.md` and `ORIGINAL_CHAT_AUDIT.md`. Registered protocols and calculator outputs: `outputs/validation/common_benchmark/`. Reusable runners: `speech_pipeline/research/common_benchmark.py` and `original_chat_audit.py`.

- All 126 corrected public test paragraphs / 49 children / 54,654 words were compared with frozen paragraph rules, previous MiniLM, latest MiniLM, and the **official Batchalign 0.10.4 CHATUtteranceBackend**, using its official word assignments/chunking and a local pinned `talkbank/CHATUtterance-en` revision `764ec3f762c2e24df2def8df98b5fe34940085c6`. Package wheel and source hashes are retained. This replaces neither the old adapter result nor the old published-system claims: it is a new official **component** comparison, not complete Batchalign audio inference or reproduction of the 2023 paper cohort.
- Same paragraph-only information for all methods. All 504 method-recording word streams were preserved and entire eligible gap masks retained. No partial-alignment scoring. All 30 predeclared native CLAN same-text checks passed.
- Corpus-balanced mean count/MLUw/population-SD APE: latest **16.0853%**, previous **17.9129%**, official Batchalign component **87.9189%**, paragraph rules **167.4538%**. Latest vs official component difference **−71.8335 points**, descriptive paired child-bootstrap interval approximately **[−79.43, −64.63]**. These are reused tests with in-domain trained MiniLM models, not independent confirmation or a fair architecture-only comparison with an untuned pretrained component.
- OCSC latest / official component: boundary F1 **85.56 / 56.31%**, metric composite **11.91 / 95.86%**. NYU latest / official component: F1 **76.92 / 49.20%**, composite **20.26 / 79.97%**. Only four NYU test children. Exact spans are appreciably worse than gap F1: latest **65.14% OCSC / 52.18% NYU**. No professional-equivalence claim.
- Also scored native CLAN on the **original fully coded main tiers for every same public task section**, not only normalized words. Latest native MLUw APE **8.74% OCSC / 17.23% NYU**; eligible-count APE **9.75 / 26.43%**; native length-SD APE **15.76 / 17.60%**. Original FREQ and our surface metric eligibility differ; lexical comparisons are explicitly policy sensitivity. Even normalized gold with human cuts differs from original native MLUw by **3.79 / 3.50%**, and FREQ TNW by **3.83 / 10.65%**. Removing coding is a real source of measurement discrepancy. Do not relabel normalized calculator agreement as professional-coded equivalence.

**Published LUCID acquisition and comparison completed, exact reproduction NOT established:** Baese-Berk et al. (2021), DOI `10.3389/fpsyg.2021.661415`; unchanged actual OSF TAALED results/R code under `dataset/public/lucid_published/`. The user completed SpeechBox’s data-use form. Use its exposed authorized sharing link, not browser cookies: `research/lucid_download.py` establishes the server-issued anonymous sharing session, then exact HTTP ranges read standard ZIP entries. All **300 TextGrids** from eight publisher archives are verified by ZIP CRC/local SHA-256; about **19,904,555 bytes** fetched. The >4 GB original L2/vocoder ZIPs failed local-header validation; officially listed split archives were used. Combined evidence: `dataset/public/lucid_published/textgrids/combined_manifest.json` (the older `manifest.json` contains only the split-archive subset). Avoid browser download-event waits: previous calls hung for 18–22 minutes. No browser credentials were extracted.

Frozen protocol/prepared inputs: `outputs/validation/lucid_validation/`. Initial preparation rejected valid empty tiers and malformed explicit noise tags; generic source-only repairs were registered and archived BEFORE scoring. No numeric results informed preparation. All **240 published rows** retained: **180 uniquely metadata-matched rows scored**, **60 vocoder rows ambiguous** (two recordings each). No favorable source choice/concatenation. `speech_pipeline/research/lucid_validation.py`; report `outputs/results/lucid_validation/REPORT.md`, complete pairs/summary/coverage/results alongside. Fifteen shared lexical measures; original MTLD is not moving-average MTLD. Existing custom and current TAALED receive identical prepared source text; unchanged production normalization. Custom mean within-speaker recording APE averaged across speakers: token counts **22.84% noBarrier / 9.78% babble / 9.63% L2**; TTR **8.65 / 3.87 / 4.24%**; MATTR50 **3.03 / 2.29 / 1.66%**; HD-D42 **2.40 / 1.37 / 0.96%**; original MTLD **14.67 / 7.51 / 7.21%**. Do not pool a four-condition score with one missing.

**Neither implementation exactly matches ANY scored row’s published token count.** This is a discrepancy audit, not exact-input reproduction or successful metric validation. The supplement matches all four article Table 1 count averages within 0.01 word and all 240 published TTRs are internally consistent with counts. Its public root has only TAALED_results.csv, TAALES_results.csv and LDLS_Analysis.R: original cleaned indexN.txt files, exact historical source/tier mapping and GUI configuration are absent. Large noBarrier count discrepancies also need mapping review; do not assume all error is implementation drift or relabel filenames by closest scores. Unsent author query: `AUTHOR_QUERY_DRAFT.md`; source audit and reproduction instructions beside it. LUCID word intervals are not professional CHAT utterances and do not validate segmentation, MLU or child-clinical diagnosis.

**First LUCID stereo audio acquired:** first published CSV row `index1.txt`, unique documented F14/babble/beach candidate. `dataset/public/lucid_published/audio/LCD_F14BEF3B4cln2.wav`, **80,701,946 bytes**, **457.478390 s**, two channels, 44.1 kHz, 16-bit PCM. Full ZIP CRC/WAV length validated; local SHA-256 and source metadata saved. Selection/acquisition protocol recorded before any LUCID generation. **No LUCID MOSS inference or audio score yet.** The OCSC real MOSS pilot below remains the only completed audio run.

**MOSS audio path and bridge implemented:** `speech_pipeline/research/moss_audio.py` runs pinned official local MOSS inference in an isolated runtime. `speech_pipeline/moss.py` imports canonical timestamp/speaker-tagged output (TXT or JSON `text`), explicitly selects the target speaker using a documented mapping, withholds MOSS segment cuts/punctuation from paragraph segmentation, preserves words, and writes faithful provisional `transcript.cha`, separate normalized CLAN metric input, lexical/clinical scores and source segment evidence. `--verify-clan` now additionally runs native CLAN on the faithful full-word CHAT, keeping that policy separate from normalized scores. Incomplete generations, absent targets and acoustic tags requiring review are rejected. No fabricated word/utterance timestamps or professional codes. Current regression suite: **78 passed**.

**First real MOSS pilot completed:** user signed into TalkBank in the in-app browser. The first saved public test task was predeclared before generation: OCSC 4002, section 5 ExpPictures, 300.021 s, original media interval 474.844–774.865 s. The bare WAV link misleadingly returned an 11-byte header after a long download; the explicit `?f=save` link succeeded. Verified full audio: `dataset/public/ocsc/media/4002.wav` (144,552,896 bytes, 1638.920159 s, mono 44.1 kHz) and a 16 kHz task excerpt under `outputs/results/moss_pilot/`.

Official OpenMOSS code `e3bdd93f8ad58418e04659818efd5fd87a0bc486`, model revision `704aa4a9c304e8520be88901e0d1960158ef5b15`, local float32 MPS inference, SDPA, default official prompt, greedy generation. Dependencies are isolated under `/private/tmp/speech_moss_runtime/dependencies`, source/model alongside; project environment unchanged. Locked 57-package requirements and raw code/model/audio hashes retained. Real inference completed EOS after **2511 generated tokens / 427.18 s**, not truncated. Its 106 segment envelopes are monotonic, with final 300.04 s within 0.05 s tolerance of the audio.

Pilot report: `outputs/results/moss_pilot/REPORT.md`; raw output, faithful CHAT, native CLAN outputs, all lexical/clinical results and errors are beside it. Protocols: `outputs/validation/moss_pilot/`. `research/moss_pilot.py` evaluates without training. **Speaker mapping is evaluation-only ORACLE minimum-edit one-to-one assignment**, CHI→S02, EXP→S01; no automatic patient identification claim. Target WER **18.8172%** (35 edits / 186 spoken reference tokens); all-speaker cpWER **14.2857%** (72 edits / 504 spoken tokens). No gold unintelligibility placeholders in this task. Native CLAN's eligible word denominator is separately **179**.

Native CLAN on full recognized-word output versus original professional CHAT: latest segmenter eligible utterances **70 vs 72 (2.78% APE)**; MLUw **2.485714 vs 2.486111 (0.01596% APE)**; length SD **0.937 vs 1.143 (18.02% APE)**; TNW **174 vs 179 (2.79%)**; NDW **71 vs 73 (2.74%)**; TTR **0.408046 vs 0.407821 (0.0551%)**; MATTR **0.541 vs 0.538 (0.558%)**. The raw MOSS source-segment baseline gives **61 eligible units**, MLUw **2.852459**, SD **1.773**: post-segmentation improves these errors on this one sample. **Close MLU partly reflects cancellation of word/count errors**, despite 18.82% WER. Do not claim perfect recognition, professional equivalence or cohort validation. Boundary F1 is intentionally unavailable on changed ASR word streams.

Precision repair: initial native MLU/TTR three-decimal displays hid small ratio errors. Before reporting, recomputed those ratios from exact native counts, with unchanged raw MOSS and frozen model predictions. Initial rounded-score evidence is archived in `outputs/validation/moss_pilot/rounded_ratio_audit/`. SD/MATTR still have native three-decimal precision. This was scoring precision repair, not model selection or favorable rerunning.

**Access resolved; remaining evidence:** TalkBank sign-in and LUCID form completion are done. LUCID source downloads work. Exact historical cleaned texts/source mapping/tool configuration remain unavailable, and vocoder rows are ambiguous. Do not ask the user to complete the form again. ENNI norms/conventions were located but ENNI was not imported/evaluated and no audio are listed.

**Next concrete steps:** (1) resolve the LUCID source/cleaning configuration, using the concrete unsent author-query draft if authorized to contact researchers; preserve the current mismatch audit; (2) freeze a multi-participant audio manifest and reviewed target-speaker mappings, exact reference policies and a bounded compute budget; (3) evaluate a compatible independent corpus/cohort before claiming confirmation. Preserve all final predictions; use public development only for any changes. One OCSC real audio pilot is complete; LUCID audio is downloaded but unscored; cohort-level audio validation and fresh confirmation remain outstanding. No ongoing job or automation was left running.

---

Updated October 15, 2026, America/Chicago. This replaces stale paused/training notes. Saved JSON/CSV values are authoritative. This update created a short user summary and current handoff; it did not start training, change selection, or launch another experiment.

## Start here

**The goal:** improve utterance segmentation AND final language measurements, then demonstrate improvement against existing pipelines on the SAME data in a conference paper. Do not substitute validation gains, arithmetic checks or lower numbers on different datasets for this goal. Explain work plainly and keep future experiments bounded; the user is frustrated by days of unclear experiments.

**What currently works best:** established rules with observed speaker turns on the original 10 participants; previous MiniLM on OCSC; latest MiniLM has lower composite metric error on NYU–Emerson. None of the evaluated learned models beats the established full pipeline on the original human values. No full published-system superiority has been demonstrated.

**User summary:** `outputs/results/project_summary/Speech_pipeline_summary.docx` is one concise document. This handoff contains technical details and next steps. No training or queue is running. Do not infer an active job from historical logs.

Prompt for a new chat:

> Read SESSION_HANDOFF.md first. Our aim is better utterances and final metrics proven against competing pipelines on common data. Rules remain best for our 10 participants; paragraph models are experimental. Use the bounded benchmark plan below before more training. Existing public test and all original participants are evaluation only and historically exposed; never tune on them. Preserve reference workbooks, frozen experiments, paper/poster/abstract. Tell me what is being compared and the time budget before substantial compute. Do not repeat closed training/selection stages merely to resume.

## What was done

1. Retained the established target-speaker workbook pipeline, which uses observed speaker turns.
2. Built joint text-plus-rule-feature paragraph models: small transformer, pretrained BERT, MiniLM, DeBERTaV3, RoBERTa and CHATUtterance initialization. Tested context, count/length objectives, corruption, balancing and weight averaging on public train/validation. Paragraphs deliberately withhold original utterance rows and speaker-turn cuts. These are different workflows from the established turn-aware pipeline, not controlled architecture-only comparisons.
3. Fixed public CHAT preparation: remove non-speech event descriptions and join explicitly coded reference continuations. Preserve spoken words; gold punctuation, rows, morphology and timestamps remain labels/reference information, not model input. Corrected spoken-word metrics are not identical to full original-coded CLAN analyses.
4. Selected `minilm_metric` on the declared public-validation metric objective. A user five-minute deadline excluded incomplete `deberta_moments` by a timestamped amendment BEFORE final access. Seven completed candidates remained, INCLUDING the incumbent. Do not say eight models completed. Latest validation objective: 9.2334 versus incumbent 17.3226. DeBERTa context had validation F1 88.09%, but was not the metric-objective selection and has no final-test result in this round.
5. Froze selection before final scoring, then re-evaluated previous MiniLM on identical corrected public inputs. Subsequently completed the new pipeline on all 37 ASR and 37 corrected-text recordings, covering all 10 participants in each condition.
6. Audited 74 native CLAN checks, all within displayed precision; source words preserved; 17 custom/TAALED comparisons per recording; all 14 frozen private predictions reproduced exactly. 63 regression tests passed. These checks establish arithmetic and reproducibility, not gold-boundary accuracy.
7. Made a direct comparison with existing manual participant values and preserved corrected-transcript software references. Publication/reference CSVs and final paper/poster/abstract were not overwritten. Found six primary-source contextual benchmarks; these do not establish a full-system win.

Serialization repair: RoBERTa's saved ByteLevel `add_prefix_space` setting was previously reset by generic loading. Fixed the loader and checked unchanged weights against in-memory validation BEFORE selection. Incorrect evidence, repair and score checks remain in the protocol. Do not restore the bad score or retrain for this repair.

## Actual results and what the numbers mean

### Ten participants and manual values

37 source recordings exist; 14 have primary manual utterance labels across all 10 participants. Reference: column E of `dataset/annotations/all_nafee_utterances.xlsx`, not the worksheet's automatic column. Pool the same labeled recordings within each participant without merging their boundary units, then take the median participant APE. Lower error is better.

| Measure | Established rules | Previous MiniLM paragraph | Latest MiniLM paragraph |
| --- | ---: | ---: | ---: |
| Total utterance-count error | **6.31%** | 38.07% | 39.00% |
| MLUw error | **6.40%** | 27.39% | 28.24% |
| Population length-SD error | **16.76%** | 22.38% | 23.64% |
| Boundary F1 on 14 recordings | **88.64%** | 76.68% | 76.46% |

The latest model oversegments all ten complete samples. Rules retain observed turns; paragraph models withhold them. Previous MiniLM WITH speaker-turn context scored 78.04% F1, also below rules. Do not change models, routing, ensembles or rules from these final errors. Labels `Previous`/`Revised` in publication files mean historical baseline/revised rules, NOT `Previous paragraph`/`New paragraph` in the latest experiment.

Separate comparison: all 37 ASR recordings versus preserved corrected CLAN/TAALED values. It includes word-transcription and reference-boundary differences. Median within-participant mean-recording errors for rules / previous paragraph / latest paragraph: eligible CLAN count **3.38 / 50.43 / 45.00%**; MLUw **3.77 / 30.81 / 29.06%**; SD **4.73 / 42.59 / 39.17%**. Eligible CLAN count differs from raw total utterance count.

### Matched public test

Assigned train/validation/test groups remain 233/49/49 children. Corrected usable train/validation groups: 231/49; two training children became unscorable and were not reassigned. Test: 49 children, 126 paragraphs, 54,654 words. OCSC contributes 45 children/111 paragraphs; NYU–Emerson 4 children/15 paragraphs. NYU evidence is consequently small.

| Dataset | Previous F1 | Latest F1 | Previous metric error | Latest metric error |
| --- | ---: | ---: | ---: | ---: |
| OCSC | **86.64%** | 85.56% | **11.46%** | 11.91% |
| NYU–Emerson | **79.39%** | 76.92% | 24.37% | **20.26%** |
| Combined | **86.22%** | 85.07% | 17.91% | **16.09%** |

F1 pools eligible word gaps. Metric error equally averages count, MLUw and population length-SD APE: recordings within child, then children within corpus, then equally across measures/corpora. Combined F1 is not equally weighted by corpus. Missing/zero-reference denominators are unavailable, not zero-filled.

The original-participant mean-recording composite slightly improved 30.31 → 29.60, although pooled complete-sample median errors above worsened versus previous MiniLM. Different pooling/aggregation explains this; neither beats rules. Paired participant-bootstrap change intervals: public −3.15 to −0.42 points; private −3.12 to +1.76. These are descriptive reused-benchmark results, not fresh confirmation. Never present one favorable aggregate as universal improvement.

Old public scores 86.07% overall / 81.85% NYU for previous MiniLM used earlier CHAT preparation. For fair latest comparisons use `metric_round.incumbent`, not those historical values. Other candidates' validation scores are NOT final-test results.

## Metrics calculated now

Available automatic transcript measures:

- Total utterances; eligible MLU utterances and eligible word counts (`mlu_nutterances`, `mlu_nwords`), with documented filler/unintelligibility exclusions.
- **MLUw:** eligible words per eligible utterance; provisional with predicted boundaries.
- **Length SD:** population SD of eligible utterance word lengths.
- **TNW:** eligible word total; **NDW:** different eligible words; **TTR:** NDW/TNW; **MATTR 50:** average TTR over moving 50-word windows, unavailable when insufficient length.
- All **17 normalized custom/TAALED lexical measures**, distinct from surface measures:
  - Six counts: `basic_ntokens`, `basic_ntypes`, `basic_ncontent_tokens`, `basic_ncontent_types`, `basic_nfunction_tokens`, `basic_nfunction_types`.
  - Two densities: `lexical_density_tokens`, `lexical_density_types`.
  - Four TTR forms: `simple_ttr_aw`, `root_ttr_aw`, `log_ttr_aw`, `maas_ttr_aw`.
  - Five diversity measures: `mattr50_aw`, `msttr50_aw`, `hdd42_aw`, `mtld_original_aw`, `mtld_mean_factor_length_aw`.

Normalized lexical values use spaCy/lemma/POS policies; surface measures have separate word eligibility. Preserve both labels. Changed grouping of unintelligible material can affect eligible words; otherwise fixed-word segmentation does not establish better lexical recognition.

Supported only with required reviewed annotations, currently UNAVAILABLE in automatic runs: **MLUm** (complete supported morphology); **WPM** (positive reviewed elapsed duration); **PGU** (complete grammaticality judgments); **omission/substitution counts** (reviewed production-error labels); **maze percentage** (reviewed maze/produced-word counts and explicit scope). Never invent these or fill missing values with zero. API: `speech_pipeline/metrics.py::clinical_measures` / `calculate_mlu`.

Evaluation measures: boundary precision/recall/F1, extra/missed cuts, exact utterance-span F1 where supported, signed/absolute metric error and APE, participant-balanced mean/median summaries and paired bootstrap intervals. Private boundaries use frozen aligned labels; private language metrics use actual human words, not human cuts imposed on ASR words. Exact private spans are unavailable without a valid shared-stream reference.

CLAN checks on the same normalized predicted text require exact counts and ratio/SD differences within 0.000501 (three-decimal display precision). TAALED preprocessing discrepancies are retained; do not claim the preprocessing pipelines always agree exactly. Human percent agreement, boundary F1, transcription WER, ICC and metric APE are distinct quantities.

## Actual benchmarks for a conference comparison

### First run competitors on common data

Use the already imported **[OCSC](https://talkbank.org/childes/access/Eng-NA/OCSC.html)** and **[NYU–Emerson](https://talkbank.org/asd/access/English/NYU-Emerson.html)** as common data. Corpus DOI: [10.21415/60TV-D669](https://doi.org/10.21415/60TV-D669) / [10.21415/VT7P-YF83](https://doi.org/10.21415/VT7P-YF83). Cite Wagner et al. (2025), *The Ohio Child Speech Corpus*, and Steen et al. (2026), *Introducing the NYU–Emerson Corpus*, per their corpus pages. They provide reference data, not competitor scores on our exact split/task.

Compare frozen rules, previous MiniLM, latest MiniLM and a pinned official **[Batchalign utterance-segmentation implementation](https://github.com/TalkBank/batchalign)** on identical inputs. Verify the selected release's supported transcript-input interface and preprocessing: current documentation changed, so old command options are not assumed valid. Keep raw outputs, version/model revisions, word edits and eligible-gap coverage. If software rewrites words, report that separately; do not quietly score only convenient aligned gaps or supply gold punctuation.

An existing local `talkbank/CHATUtterance-en` checkpoint ADAPTER scored 67.01% versus previous MiniLM 86.07% on the same HISTORICAL prepared public inputs. It omitted parts of official inference and has not been rechecked on corrected inputs. It is a matched adapter baseline, NOT a reproduction of full Batchalign or evidence of beating its published result. Rerun corrected inputs before comparing with latest models.

Separate permitted input conditions: A) unpunctuated target-speaker paragraph, no source-turn cuts; B) actual observed speaker changes/turns. Within B, merge consecutive same-speaker gold rows so utterance-row cuts do not leak. All methods get the same information within a condition. Do not compare turn-aware rules with paragraph models as an architecture-only ablation.

Publish per-corpus sample counts, input condition, pinned method, boundary P/R/F1 and exact spans, count/MLUw/SD errors, TNW/NDW/TTR/MATTR errors, missing counts and paired participant intervals. Fix reference units, exclusions, primary metric and aggregation beforehand. Existing exposed tests support descriptive comparison; independent confirmation needs new data.

### Independent datasets to consider next

- **[ENNI](https://talkbank.org/childes/access/Clinical-Eng/ENNI.html)**, DOI [10.21415/T51G7V](https://doi.org/10.21415/T51G7V): 361 included children aged 4–9; picture-generated narratives including language impairment. Candidate child-language external validation, not imported/tested. Its gold units are **C-units**, not automatically our CHAT target; check conventions, contractions and terms before using. The corpus lists media as unavailable, so do not promise audio evaluation. Source: Schneider et al. (2006), linked on the corpus page.
- **[Santa Barbara Corpus of Spoken American English](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english)**: spontaneous conversations, transcripts/audio/timestamps and CHAT versions. Optional conversational-domain dataset, not imported/tested. Its intonation units are not automatically compatible utterances: establish suitable references or blinded reannotation and check usage terms. It is not a ready-made CHAT-unit gold standard.
- For the original deployment population, newly recruited participants with independently adjudicated labels offer stronger confirmation. More recordings from the original ten are not new independent participants.

Choose a new corpus from population/task/reference fit BEFORE viewing performance, never to obtain a winning result. TalkBank access may require authorized sign-in; no credentials are stored. This document does not authorize collecting or importing new data silently.

### Published numbers are context only

| Study | Published result | Correct interpretation |
| --- | --- | --- |
| [Liu et al. 2023 Batchalign](https://pubs.asha.org/doi/full/10.1044/2023_JSLHR-22-00642) | Adult Delaware MCI F1 86.9%; controls 85.0% table / 85.1% prose | Different recordings/task; no direct ranking against our child-corpus scores. |
| [Pelfrey et al. 2024 IDEAS Table 4](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1322665/full) | Children's own-talk median count error 11.11%, word-MLU error 9.29%; unique words 29.17%, words 30.00%, TTR 17.43% | Includes ASR and six-minute segment aggregation; different from fixed-word paragraphs. Reproducible common-data baseline feasibility is unestablished. |
| [Maltman et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10023176/) | Human segmentation agreement 83.21%, 11 independently retranscribed samples | Reliability context; disagreement complement is not a known true error rate or boundary F1 ceiling. |
| [MacFarlane et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10123085/) | Human segmentation agreement 82% | Audio-informed C-unit agreement, not comparable to our word-gap F1. |

All six sourced studies, including ICC and human-WER references, are in `outputs/validation/benchmark/literature_benchmarks.json`. IDEAS has NOT been run on OCSC or NYU; lower OCSC errors do not demonstrate beating IDEAS. Never convert 1−ICC or 100−agreement into our APE. To claim human-level behavior, use blinded raters, compatible units, independent adjudication and identical human–human/system–human scoring on the same input.

## Next experiment order

1. Build the bounded **common-input benchmark harness before more training**. Reuse frozen models, pin external implementation, evaluate same corrected examples and save raw predictions, hashes, gap masks, word edits, per-child errors and runtime. Report losses as well as wins.
2. Audit PUBLIC TRAIN/VALIDATION only to justify a change. Candidate hypothesis: joint text/rule model that uses actually available speaker-turn context, or development-supported training changes addressing extra boundaries and language metrics. Existing turn-transfer failed, so preserving turns alone is not a solution. No private/final-label-driven model design, hard count forcing, arbitrary utterance caps or posthoc cutoffs.
3. If development supports it, register a small text-only/rules-only/joint ablation under MATCHED context and participant-disjoint groups. More/larger encoders alone did not win the last objective. Freeze choices before external confirmation. Do not reopen old selection or reshuffle exposed test groups.
4. Confirm once on a compatible independent corpus or new participants. Predeclare endpoints and paired participant analysis, include all eligible records, and scope claims to supported tasks/metrics. The goal is improvement; it is not permission to manufacture a favorable benchmark.
5. Before spending compute, tell the user the comparison, purpose and time budget. Afterward report clearly whether the main goal improved. No unexplained days-long queues or automatic open-ended continuations.

## Evidence and continuation map

Workspace: `/Users/nafeekazi/Desktop/PhD/speech_pipeline_code`; research runtime `.venv/bin/python`. MPS access may require outside-sandbox execution; CPU fallback is supported. No subagent work was used for these experiments.

| Purpose | Path |
| --- | --- |
| User summary | `outputs/results/project_summary/Speech_pipeline_summary.docx` |
| Latest ten-participant report | `outputs/results/participant_pipeline_comparison/REPORT.md` |
| Each human/pipeline value and APE | `outputs/results/participant_pipeline_comparison/participant_values.csv` |
| Per-measure pairs/summaries/boundaries | Same folder: `metric_pairs.csv`, `metric_summary.csv`, `boundary_recordings.csv`, `recording_manifest.csv` |
| Preserved previous full-run summaries | Same folder: `previous_paragraph_uncorrected.json`, `previous_paragraph_corrected.json` |
| Latest full outputs | `outputs/results/pipeline/{uncorrected,corrected}/` |
| Frozen latest results plus matched incumbent | `outputs/results/benchmark/improved_model.json::metric_round` |
| Prior locked round | Same file `::reassessment`; `outputs/validation/benchmark/reassessment.json` |
| Latest protocol/history/selection/deadline amendment | `outputs/validation/benchmark/metric_training.json` |
| Main report with historical sections | `outputs/results/benchmark/REPORT.md` |
| Original human/software references | `outputs/results/human_comparison/` |
| Publication tables and audit | `outputs/results/paper/`, `outputs/validation/audits/error_summary/` |
| Public identities and split | `dataset/public/participants.csv`, `outputs/validation/benchmark/talkbank_split.json` |
| Corrected public data | `outputs/validation/benchmark/talkbank_spoken_development.jsonl`, `talkbank_spoken_test.jsonl` |
| Frozen private aligned data and sources | `outputs/validation/benchmark/boundary_dataset.jsonl`, `boundary_dataset_sources.json` |
| Literature | `speech_pipeline/research/SEGMENTATION_RESEARCH.md`, `outputs/validation/benchmark/literature_benchmarks.json` |

**Routing:** default `--boundary-context speaker-turns --segmenter auto` uses established rules; explicit paragraph context uses the latest learned bundle. Canonical outputs currently reflect an explicit EXPERIMENTAL paragraph run, not default rules. Use separate output roots to preserve comparisons.

Bundles under `outputs/results/benchmark/`: `metric_segmenter/` latest, `context_segmenter/` previous MiniLM, `segmenter/` earlier BERT. Latest weights SHA-256 `ad8f44b88979572231ff59e886d0e8464b15a2d6d9037841d2e2b23dd861ebbb`; previous MiniLM `9f90262f4154c2261034872153b9c7cde9040cca892fd99f763ca532265a9b31`. Latest fixed argmax, no tuned threshold or hard utterance limits. Three bundle files: model.safetensors, tokenizer.json, segmenter.json. Candidate caches are temporary under `/private/tmp/speech_metric_research`; interrupted context evidence is `interrupted_context_handoff/`. Do not assume temporary weights persist indefinitely.

Corrected development SHA-256 `d66d60ce3ff0d665d7f459e2fecbf1a2254a674ca6cfac5a92ee520da527242b`; private dataset SHA-256 `564ac5484949da9964dd0b5e0d28e6572d16b7fe3a2c72bf2217c361e65cc0e1`. Protocol contains test hashes, selection-before-final timestamps, reload repairs and histories. `software.json::metric_round` contains package/source/publication hashes for completed runtime verification. Later documentation edits do not imply retraining.

Safe commands when needed:

```sh
# Rebuild ten-participant comparison from frozen predictions/references
.venv/bin/python -m speech_pipeline.research.participant_pipeline_comparison
# Refresh report without training
.venv/bin/python -m speech_pipeline.research.metric_training report
# Code checks
.venv/bin/python -m unittest discover -s tests
# Default rules on a separate output root
.venv/bin/python -m speech_pipeline --metric-profile clinical --verify-clan --output-root /private/tmp/speech_rules_check
```

Canonical paragraph run and `metric_training runtime` are COMPLETE. Separate all-public-final lexical scoring stage `metric_training downstream` remains PENDING; this does not invalidate completed count/MLUw/SD or ten-participant results. Inspect before running it as frozen scoring if requested. Do not rerun `fetch-models`, `register`, `prepare`, training, `select` or final `evaluate` to resume. Never remove locks to choose from observed final outcomes.

Preserve `conference_paper/conference_paper.docx`, poster PDF/PPTX and abstract in `thompson_center_poster/`, source workbooks and original paper/manual CSVs. No publication edits are authorized by this documentation task. This handoff replaces obsolete prose; it does not alter immutable experimental evidence.
