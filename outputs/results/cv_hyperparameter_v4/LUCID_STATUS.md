# LUCID: published lexical comparison

**Publication provenance check pending:** current OSF access fails. The retained
reference CSV matches saved acquisition metadata; an independent remote copy or
author confirmation is still needed to authenticate its provenance for publication.
Results are preserved and provisional for that purpose; no values are invented,
replaced, or inferred from the paper's figures.

Reporting clarification:15 compared columns come from the authors' released CSV;
the main article does not present all15. Original MTLD is an additional export
field and is not the moving-average variant in Figure4. See
../transcript_validation/LUCID_METRIC_PROVENANCE.md for the complete column/source
mapping, saved OSF hash verification and current remote-access limitation.

The reference is the authors' actual per-sample `TAALED_results.csv`, accompanying
[Baese-Berk et al. (2021)](https://doi.org/10.3389/fpsyg.2021.661415).
This assesses reproduction of published lexical statistics; the paper does not
provide a competing boundary model's accuracy, gold utterances or MLU references.

| Condition | Scored / published samples | Word-count mean participant APE % | MATTR50 mean participant APE % |
|---|---:|---:|---:|
| Babble | 60 / 60 | 0.76 | 0.44 |
| L2 | 60 / 60 | 1.87 | 0.44 |
| No barrier | 60 / 60 | 16.60 | 2.11 |
| Vocoder | 0 / 60 | Unresolved | Unresolved |

All fifteen shared measures meet the declared mean-error engineering targets for
babble and L2. No barrier still fails several measures. Vocoder's source mapping
remains ambiguous; those sixty samples are unavailable, not zero-error results.
Coverage retains all240 published rows and all300 source TextGrids. Exact historical
cleaned inputs and source/tier mapping remain unresolved. No recording is chosen by
its numeric agreement with the published score.

An additional source-only audit found no contradiction between available speaker
comments and numeric Speaker A/B tier names. Comments inherited by mixed, keyword
or unnamed tiers do not establish a speaker-label error or resolve target identity.
See `outputs/validation/cv_hyperparameter_v4/lucid_comment_tier_summary.json`.

These lexical results are unchanged by the current boundary-model training search.
LUCID supplies no training examples or model-selection folds. Full retained results,
per-sample pairs, coverage and calculator checks are in
`outputs/results/transcript_validation/lucid_results.json` and `REPORT.md`.
