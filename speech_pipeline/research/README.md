# Paper and poster analysis

## Current public benchmark

See `outputs/results/common_benchmark/REPORT.md` for the official Batchalign 0.10.4 component comparison on 126 identical public paragraphs. `ORIGINAL_CHAT_AUDIT.md` measures sensitivity to original coded CHAT using native CLAN on every same task section. The ten original participants were excluded from this continuation. No training or frozen model choice was changed.

The new benchmark is registered in `outputs/validation/common_benchmark/protocol.json`; do not delete the protocol or rerun final comparisons to tune a system. `common_benchmark report` refreshes presentation from retained results. The original-CHAT audit is a separate reference-policy analysis, not a replacement primary endpoint.

For independent exact-value lexical validation, Baese-Berk et al. (2021), DOI `10.3389/fpsyg.2021.661415`, release actual TAALED results and R code at their cited OSF view-only link. These are saved under `dataset/public/lucid_published/` with hashes. The user completed SpeechBox’s data-use form. All 300 TextGrids were selectively downloaded from the authorized archives, with ZIP CRC and local SHA-256 checks. The first published sample’s full stereo audio is also retained. `research.lucid_validation` prepared and scored all 180 uniquely metadata-matched rows across 15 shared measures; all 60 vocoder rows remain unavailable because each key has two candidates. Results: `outputs/results/lucid_validation/REPORT.md`. Custom token-count APE is 22.84% noBarrier / 9.78% babble / 9.63% L2; no exact token-count matches. This is a discrepancy audit, not successful exact reproduction. Original cleaned indexN.txt inputs, source/tier mapping and historical GUI configuration are still needed. See `outputs/validation/lucid_validation/REPRODUCE.md` and `AUTHOR_QUERY_DRAFT.md`. No LUCID MOSS run has been performed.

Runtime support for MOSS output ingestion is `python -m speech_pipeline.moss --help`; it requires an explicit target mapping and preserves source segment timing separately. Real audio inference is `python -m speech_pipeline.research.moss_audio --help`, in an isolated official MOSS environment. The predeclared first OCSC task pilot is complete: `outputs/results/moss_pilot/REPORT.md`. It uses oracle speaker-role matching for evaluation and provides one-sample feasibility evidence. Do not represent it as cohort validation or exact published-score replication. Scoring is `research.moss_pilot`; existing outputs refuse overwriting. MLU/TTR reference ratios are recovered from exact native counts, while native SD/MATTR retain displayed precision.

## Metric-first development round

The resumed round is recorded in `outputs/validation/benchmark/metric_training.json`; current operational status is at the top of `SESSION_HANDOFF.md`. Its objective minimizes corpus-balanced participant errors in utterance count, surface MLUw and population utterance-length SD. Boundary F1 remains secondary. Corrected spoken-word development data and the child assignments are frozen. Selection is now frozen: a user deadline amendment excluded the incomplete moments candidate before final access, leaving seven completed candidates. The original participants and public test remain evaluation only. The full selected-model canonical run and ten-participant comparisons are complete; see `outputs/results/participant_pipeline_comparison/REPORT.md`.

Run individual stages with `.venv/bin/python -m speech_pipeline.research.metric_training STAGE`. The remaining candidate order is `deberta_context`, `deberta_metric`, `roberta_metric`, `minilm_metric`, `deberta_sentence`, `chat_metric`, `deberta_moments`; the incumbent is already complete. Completed candidates refuse retraining. Do not run `fetch-models`, `register` or `prepare` again on the retained round. The interrupted context checkpoint/history were archived and the incomplete candidate restarted from the pinned initialization with unchanged seed/settings; optimizer-state resume was not available.

After every registered candidate completes, run `select`, `evaluate` and `downstream` in that order. Selection closes development and copies only its chosen bundle into `outputs/results/benchmark/metric_segmenter/`. Evaluation compares the choice with the frozen incumbent on identical corrected inputs. Then regenerate canonical outputs with `.venv/bin/python -m speech_pipeline --boundary-context paragraph --metric-profile clinical --verify-clan`, followed by the research stages `runtime` and `report`. A `runtime` audit requires that canonical run; it does not generate it. No final result licenses model, loss, rule, threshold or routing changes. Local Apple GPU access may require running outside the restricted sandbox; CPU fallback is supported.

Published automated and human comparisons are in [the research review](SEGMENTATION_RESEARCH.md#human-comparisons-and-metric-first-benchmarks-verified-october-15-2026). Exact sourced values, units, aggregation details and comparison limits are retained in `outputs/validation/benchmark/literature_benchmarks.json`. Human percent agreement, consistency ICC, WER, boundary F1 and language-metric percentage errors must remain distinct.

The final retained manuscript is `conference_paper/conference_paper.docx`. Its author-edited wording, figures, comments, and structure are preserved. The author designated its editing baseline on October 9, 2026, with an eight-page target. The October 8 reference and accuracy audits predate those author edits; they are not a fresh audit of every sentence in the current document.

## Refresh publication outputs

Run from the project root, using the preserved audited scores:

```sh
.venv/bin/python -m speech_pipeline.research.publication_data
.venv/bin/python -m speech_pipeline.research.median_error_analysis
.venv/bin/python -m speech_pipeline.research.render_nafee_results
```

Outputs and saved figure data are in `outputs/results/paper/`. Run the median stage after rebuilding publication data. Rendering needs the `figures` dependency extra and Poppler; `--manual-only` refreshes just the primary-human error figure. Existing system-design, boundary-example, and WER assets remain in the paper results.

## Recompute upstream scores

These commands write new scores when you intend to rescore the research inputs:

```sh
.venv/bin/python -m speech_pipeline.research.import_clan_results
.venv/bin/python -m speech_pipeline.research.compare_utterances
.venv/bin/python -m speech_pipeline.research.nafee_sample_analysis
.venv/bin/python -m speech_pipeline.research.transcript_wer
.venv/bin/python -m speech_pipeline.research.human_interrater
```

Human annotation inputs are in `dataset/annotations/`; paired scores are in `outputs/results/human_comparison/`. Saved predictions and the original baseline are in `outputs/validation/method_comparison/`. `baseline_utterances.py` preserves the historical implementation. `render_slide_figures.py` regenerates the supporting presentation figures.

`outputs/validation/audits/error_summary/mean_median_analysis.json` records exact original source-file hashes. The median stage resolves their historical paths and verifies the unchanged files. Upstream rescoring can change bytes, including insignificant floating-point differences in TAALED's HD-D output. Changed scores require review and a refreshed audit; this provenance check remains enforced.

## Interpretation

The main human comparison uses column E (Manual utterance) for 14 sampled recordings from all 10 participants. Worksheet column D is not the automatic prediction source. Complete samples use one paired absolute percentage error per participant; recording comparisons use the median of within-participant mean recording errors. Current tables and intervals are `outputs/results/paper/publication_tables_median.csv` and `median_robustness_analysis.json`. Original mean summaries remain for the audit.

`human_interrater_results.csv` is human-to-human agreement with fillers excluded. The historical `interrater_results.csv` contains pipeline-to-rater comparisons and must not be presented as human-to-human reliability. The selected-reference experiment in `outputs/validation/reference_experiment/` is excluded from the current manuscript.

Source-line corrected references, automatic-boundary corrected references, human labels, identical-input calculator checks, and human interrater agreement are distinct analyses. The paired CLAN inputs and official outputs are in `outputs/validation/clan/`. Regenerate the automatic-boundary export with `.venv/bin/python -m speech_pipeline.clan`; run CLAN externally and retain each input with its corresponding output.

## Paragraph segmentation research

The [primary-source review](SEGMENTATION_RESEARCH.md) records the techniques, real related papers, published results and differences in evaluation. Production inference is in `speech_pipeline/segmentation.py`. The latest fixed-decoder experiment is `research/reassessment.py`; `research/segmentation.py` preserves the earlier BERT round. The root README describes runtime commands.

Research, plotting and test dependencies:

```sh
uv sync --locked --extra figures --extra public-data --extra pretrained
```

The latest evidence is `outputs/validation/benchmark/reassessment.json`. It records the existing public child partitions, candidates, training history, model revisions, compute interruption, validation-only selection and final-evaluation timestamp. This round is complete and locked: training and candidate changes are refused after selection. Candidate weights and caches remain under `/private/tmp/speech_reassessment/`; only the selected production bundle was copied into `outputs/results/benchmark/context_segmenter/`. The previous selected bundle and random-initialization checkpoints remain as reproducible comparisons.

The implemented stage order is `fetch-models`, `register`, `incumbent`, `bert_context`, `minilm_context`, `minilm_sequence`, `select`, `evaluate`, `downstream`, `report`. It documents the experiment; do not delete its lock or repeat selection using final scores. `fetch-models` downloads exact pinned official revisions into temporary storage, without remote repository code. If reproducing independently, preserve this evidence and use a separate copy and output location. Local CPU/GPU numerical differences may prevent byte-identical weights.

The retained frozen predictions can refresh the existing report without retraining:

```sh
.venv/bin/python -m speech_pipeline.research.reassessment report
```

`downstream` scores all frozen test predictions with the existing metric policy and verifies six distinct-child public plus all 14 private paragraph inputs with actual CLAN, including length SD. It appends `method=reassessed` in the existing metric CSV and retains prior rows. The full production paragraph run processes all 37 recordings in each condition and stores 74 CLAN checks and 17 TAALED comparisons per recording. Only 14 recordings across the 10 participants have the human boundary labels used here. Current canonical outputs reflect this explicit paragraph run; the application default for speaker-turn workbooks remains established rules, as fixed before final evaluation. No publication artifact is edited.

For historical random-initialization inference:

```sh
.venv/bin/python -m speech_pipeline.boundary_model segment --model outputs/results/benchmark/hybrid_model.pt --input /path/to/paragraph.txt
```

The original 10 participants are final evaluation only going forward. Do not train, tune rules, choose thresholds/epochs/models or change routing using them. Public train/validation children determine every choice. Both final benchmarks were inspected historically, and the inherited rules were developed earlier on the original cohort; neither benchmark is pristine. Repeated testing and reshuffling cannot establish fresh independent validation. All scores are transcript segmentation results, not new audio/ASR results, SALT equivalence or clinical certification.
