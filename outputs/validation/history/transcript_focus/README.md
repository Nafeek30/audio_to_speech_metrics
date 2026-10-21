# Speech analysis project

# Public common-input benchmark and MOSS output import (October 15)

The active conference-paper scope excludes the original ten-participant cohort. New same-public-data results are in [the matched benchmark report](outputs/results/common_benchmark/REPORT.md), [the original-coded CHAT audit](outputs/results/common_benchmark/ORIGINAL_CHAT_AUDIT.md), and [the first real MOSS audio pilot](outputs/results/moss_pilot/REPORT.md). The official Batchalign 0.10.4 segmentation component is compared with frozen models on exactly the same 126 paragraphs. One preselected five-minute OCSC audio task has also completed MOSS-to-CHAT/native-CLAN evaluation with an explicitly oracle speaker-role mapping. The [published LUCID comparison](outputs/results/lucid_validation/REPORT.md) now scores 180 metadata-matched samples across 15 shared metrics; 60 vocoder rows remain ambiguous. All 300 source TextGrids and the first published sample’s stereo audio are downloaded. Significant count discrepancies remain, so exact published-value reproduction and multi-participant audio validation are still outstanding.

Canonical MOSS timestamp/speaker-tagged output can now be imported:

```sh
.venv/bin/python -m speech_pipeline.moss /path/to/moss_output.json \
  --output /path/to/new_output_folder --target-speaker S02 \
  --mapping-note 'Describe the reviewed evidence identifying S02 as the target' \
  --verify-clan
```

The importer produces a faithful provisional CHAT transcript that retains recognized fillers/repetitions, a separate normalized calculator input, metric results, and original segment envelopes. With `--verify-clan`, native CLAN also scores the faithful recognized-word CHAT separately. The speaker ID must be explicitly identified; it is not inferred from speech quantity. MOSS segment timestamps are not fabricated into timings for newly split utterances. Audio inference is provided by `speech_pipeline.research.moss_audio`, using the [official MOSS runtime](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) in a separate environment. Its Transformers 5 requirement conflicts with this project's frozen Transformers <5 environment; pinned pilot dependencies and reproduction instructions are retained under `outputs/validation/moss_pilot/`.



Latest metric-first round: the selected paragraph bundle is now `outputs/results/benchmark/metric_segmenter/`. Final composite count/MLUw/length-SD error improved from **17.91% to 16.09%** on public test and **30.31% to 29.60%** on the original participants. Boundary F1 declined to **85.07%** and **76.46%**, respectively. Selection used public validation only. The subsequent 10-participant comparison completed all 74 canonical recording-condition outputs and native CLAN/TAALED checks. Against the existing manual participant values, established rules remain substantially better: median count/MLUw/SD errors are 6.31/6.40/16.76% versus 39.00/28.24/23.64% for the new paragraph model. See [participant comparison](outputs/results/participant_pipeline_comparison/REPORT.md). The earlier composite and this complete-sample median analysis use different aggregation. See [latest results](outputs/results/benchmark/REPORT.md) and [handoff](SESSION_HANDOFF.md). The new round is closed to further tuning on these final results.


For a new chat, start with [SESSION_HANDOFF.md](SESSION_HANDOFF.md): it contains the complete session report, results, research sources, file map, and continuation restrictions.

Run from this folder:

```sh
.venv/bin/python -m speech_pipeline
```

The pipeline starts from the existing transcript workbooks and corrected text for 37 recordings. It selects the mapped speaker, segments diarized workbook speech with the established rules, calculates custom and TAALED measures, and exports CHAT inputs for CLAN scoring. Add `--metric-profile clinical --verify-clan` for the requested metric profile and actual installed CLAN comparisons. New runs go to `outputs/results/pipeline/`.

For a fresh environment, use Python 3.11–3.13 and `uv sync --locked --extra segmenter`. The selected trained segmenter is stored locally and loads without network access. Figure generation additionally needs the `figures` extra and Poppler. The existing `.venv/` is ready to use.

## Where things are

| Folder | Purpose |
| --- | --- |
| `conference_paper/` | The final retained conference paper, `conference_paper.docx`. |
| `thompson_center_poster/` | Final poster PDF, matching editable PowerPoint, and abstract. |
| `speech_pipeline/` | Runnable code. Paper and poster analysis helpers are in `research/`. |
| `dataset/` | Original videos, transcripts, speaker mapping, corpus resources, and human annotation workbooks in `annotations/`. |
| `outputs/results/` | Results to inspect: `paper/` figures and tables, `human_comparison/` paired scores, `presentation/` figures, and `target_transcripts/` annotation exports. New pipeline runs create `pipeline/` here. |
| `outputs/validation/` | Supporting evidence: `clan/` inputs and official outputs, `method_comparison/` saved predictions, `original_analysis/` baseline reports, `audits/` integrity checks, `comparisons/` historical workbooks, and `reference_experiment/` historical reviewer analysis. |
| `tests/` | Segmentation, scoring, annotation, and corrected-turn regression tests. |

The final paper, poster, and abstract retain their original content. Earlier document versions and templates have been removed.

Folder names describe their purpose. Recording filenames retain the original recording date/time and participant identifiers so transcripts can be traced to the correct video. CLAN input filenames retain content hashes where they identify the exact input used by a saved official result. Those identifiers are part of the research evidence. Dates inside audit records and random seeds are preserved, while directory and audit-file names have been simplified.

## Commands

```sh
# Process one transcript version
.venv/bin/python -m speech_pipeline --dataset uncorrected

# Import CLAN outputs for that new run after running MLU externally
.venv/bin/python -m speech_pipeline --dataset uncorrected --clan-results outputs/results/pipeline/uncorrected/clan_software_input_files

# Re-export the four canonical CSV tables per dataset
.venv/bin/python -m speech_pipeline.export --output-root outputs/results/pipeline

# Export original mapped-speaker rows for annotation
.venv/bin/python -m speech_pipeline.export --manual-samples

# Rebuild the historical comparison workbook
.venv/bin/python -m speech_pipeline.export --revision-workbook outputs/validation/method_comparison/utterance_comparisons.xlsx

# Regression checks
.venv/bin/python -m unittest discover -s tests -v
```

Paper-analysis commands and statistical definitions are in `speech_pipeline/research/README.md`. CLAN commands and the distinction between the two reference procedures are in `outputs/validation/clan/README.md`.

Saved research reports retain their original content and hashes. The code resolves historical paths recorded in their audit metadata to the simplified layout. Upstream rescoring can change exact output bytes; the original manuscript audit must be reviewed before publishing newly rescored results.

## Public CHAT validation and individual results

The current experiment report is `outputs/results/benchmark/REPORT.md`. It includes recording-level plots, the requested metric profile for the existing recordings, and a fresh official CLAN/TAALED calculator comparison. The paragraph segmentation transformer trains on the imported OCSC and NYU–Emerson CHAT corpora, with separate children reserved for validation and testing. Supporting sources and software hashes are in `outputs/validation/benchmark/`. The final conference paper and poster have not been edited.

```sh
# CHAT parsing and plotting dependencies
uv sync --locked --extra figures --extra public-data --extra pretrained

# TalkBank requires sign-in. Download the authorized transcript ZIP in your browser,
# then import it; no password or browser cookie is stored in this project.
.venv/bin/python -m speech_pipeline.corpora import-zip nyu_emerson /path/to/NYU-Emerson.zip
.venv/bin/python -m speech_pipeline.corpora import-zip ocsc /path/to/OCSC.zip

# Inspect speaker codes, tasks, morphology, timing, error marks, and linked media
.venv/bin/python -m speech_pipeline.corpora inspect nyu_emerson
.venv/bin/python -m speech_pipeline.corpora inspect ocsc

# Transcript-only external validation; CHAT answer punctuation is withheld.
# Check speaker/task names in the inventory before selecting them.
.venv/bin/python -m speech_pipeline.corpora benchmark nyu_emerson --speaker CHI
.venv/bin/python -m speech_pipeline.corpora benchmark ocsc --speaker CHI --gems Howto DescriptivePictures ExpPictures

# Existing-cohort recording-level plots and requested metric tables
.venv/bin/python -m speech_pipeline.research.validation analyze

# New pipeline runs with the requested clinical-style metric profile
.venv/bin/python -m speech_pipeline --metric-profile clinical
```

Authorized corpus files go under `dataset/public/nyu_emerson/` and `dataset/public/ocsc/`. Matching downloaded audio/video may be kept in each corpus's `media/` subfolder with its original filename. The corpus registry contains the official media links and required citations. Empty media folders and fake download files are not created. Audio ASR/diarization evaluation is a separate stage from the transcript-only benchmark.

The profile computes **MLUw, TNW, NDW, TTR, and MATTR-50** under the stated surface-word policy. **MLUm, WPM, PGU, language-production omissions/substitutions, and maze percentage** remain missing unless the required supported morphology, reviewed timing, grammaticality, error, and maze annotations are supplied. These surface measures differ from TAALED's lemma/POS normalization. They are not automatically SALT-equivalent.

For explicit reviewed annotations, call `clinical_measures` in `speech_pipeline/metrics.py`, or run:

```sh
.venv/bin/python -m speech_pipeline.metrics /path/to/annotations.json --output outputs/results/benchmark/reviewed_metrics.json
```

Its JSON accepts `reviewed`, `utterances` with explicit main-body `words`, optional aligned `mor` with `morphology_scheme: "legacy_mor"`, explicit boolean `grammatical` labels, and explicit `omission_count`/`substitution_count`. WPM requires positive `duration_seconds` and `timing_reviewed: true`. Mazes require an explicit `maze_scope`: `whole_sample` uses reviewed `maze_words` and `produced_words` for every utterance, including excluded ones; `eligible_utterances` uses the explicitly selected main-body/maze counts. Missing codes are never treated as evidence of grammaticality or zero errors.

The experiment uses official compiled CLAN programs installed as `.venv/bin/clan-mlu` and `clan-freq`. Their source/build hashes are in `outputs/validation/benchmark/software.json`. CLAN `freq +k` matches case folding; comparison of TAALED calculators on identical normalized tokens is separate from comparison of the two preprocessing pipelines.

## Selected paragraph model

The new model combines a pretrained 12-layer MiniLM encoder with learned rule features at each word gap. It uses both surrounding words, overlapping windows, and fixed class argmax. No probability cutoff, hard utterance-length limit or pause threshold is selected. Training-only crops vary the available context; an equal parameter average of the final three epochs won on validation. A learned CRF sequence decoder was tested but did not improve validation over this model. These experiments do not isolate the benefit of each component.

The previous selected local bundle is three files in `outputs/results/benchmark/context_segmenter/`; the latest selection is `metric_segmenter/` beside it. The previous four-layer BERT bundle in `segmenter/` is retained for reproducible comparisons. Other new candidate weights and training caches stay in temporary storage outside the project.

```sh
# Runtime dependencies for the selected model
uv sync --locked --extra segmenter

# Join target speech into one paragraph per recording, then segment and score.
# This is the mode used for the latest canonical outputs, all 37 recordings
# in each transcript condition.
.venv/bin/python -m speech_pipeline --boundary-context paragraph --metric-profile clinical --verify-clan

# A new plain-text paragraph: segments, TAALED, surface measures and CHAT
.venv/bin/python -m speech_pipeline --paragraph /path/to/paragraph.txt --metric-profile clinical --verify-clan

# Established workbook method with observed speaker turns (application default)
.venv/bin/python -m speech_pipeline --metric-profile clinical --verify-clan
```

`--verify-clan` requires the installed official `clan-mlu` and `clan-freq` programs. Without it, the pipeline exports CHAT for external scoring. `--segmenter improved` explicitly requires the trained bundle; with the default speaker-turn context it runs the experimental transfer condition. `auto` uses the model for paragraphs and established rules for speaker-turn workbooks. That behavior was fixed before this round's final scores. The latest explicit paragraph run replaces canonical generated outputs; it does not change the application default. Historical publication scores remain separate.

The clinical CSV includes custom values, CLAN values and their differences. All 17 custom/TAALED comparisons remain in the lexical CSV. Calculator agreement on identical predicted text does not establish correct utterance boundaries or clinical annotation.

Training uses **both OCSC and NYU–Emerson**, with the existing child-disjoint **233/49/49 train/validation/test split**. Across all partitions there are 331 usable children, 907 paragraphs and 347,502 words. Original punctuation, sentence casing, row boundaries, speaker-turn cuts, morphology and timestamps are withheld from paragraph model inputs. Gold CHAT boundaries are labels only; final and unintelligible-adjacent gaps are masked in boundary scoring. Window and crop positions never depend on gold boundaries. The original source spelling and all words are retained in output segments.

This round improved matched paragraph boundary F1 from **83.34% to 86.07%** on the public test and **72.88% to 76.68%** on the original 10 participants. The latter uses 14 human-labeled recordings, one target-speaker paragraph per recording. The secondary speaker-turn model result is **78.04%**, below the paper's established rules at **88.64%**. It therefore does not establish an improvement over the published full pipeline. Detailed metric errors, uncertainty and all participant/corpus results are in the [results report](outputs/results/benchmark/REPORT.md). Real related papers and the limits of cross-paper comparisons are in the [research review](speech_pipeline/research/SEGMENTATION_RESEARCH.md#reassessment-from-first-principles).

`improved_model.json` beside the report retains earlier results and the new `reassessment` section. `improved_metric_errors.csv` contains every comparison, with the new model identified by `method=reassessed`. Supporting evidence is in `outputs/validation/benchmark/reassessment.json` and `software.json`. Full research dependencies and commands are in `speech_pipeline/research/README.md`.

The frozen data and child manifest are `outputs/validation/benchmark/talkbank_training.jsonl` and `talkbank_split.json`. The reviewed identity CSV uses `corpus,recording,speaker,participant,verified`; every session of a child uses the same verified participant ID. Do not regenerate inputs or reshuffle children to improve test scores.

The original **10-participant cohort is final evaluation only going forward**: no training, adaptation, threshold/epoch selection, rule refinement or model/ensemble/routing selection. Every decision in this round was frozen using public training/validation before opening the final records. The cohort, public test and inherited rules have historical use, so these are **reused research benchmarks**, not untouched confirmatory tests. An independent cohort is needed for a confirmatory claim. Final conference paper, poster and abstract remain unchanged.
