# LUCID metric provenance and reporting correction — 2026-10-16

## Amendment — remote reference verified on 20 October 2026

The paper-linked OSF CSV was independently retrieved successfully on 20 October
2026 (HTTP 200, curl exit 0). Its 94,436 bytes, 240 rows and SHA256
`f4e9ad97e50d859311f04502a2c462826a783b064a5e4ce6e4282b058b947b63`
match the retained reference exactly. This resolves the remote-byte provenance
qualification below. The earlier failed-access account is preserved as history.
Source copy, response headers and receipt:
`outputs/research/published_metric_values_2026-10-20/sources/TAALED_results.csv`,
`outputs/research/published_metric_values_2026-10-20/sources/osf_response_headers.txt`,
`outputs/research/published_metric_values_2026-10-20/source_verification.json`.

This amendment does not resolve cleaned-input mapping, vocoder source ambiguity,
historical tool settings, the original-versus-moving-average MTLD distinction,
or the 0.01 vocoder token-mean discrepancy. No model was run or rescored.

## Historical 16 October status (superseded for remote-byte verification)

**Publication status: independent author-source reconfirmation pending.** The
project retains the CSV, source URL/hash record, an OSF-shaped API listing and the
analysis script. Their mutual consistency supports the recorded provenance but
does not independently authenticate a historical HTTP transfer. The project-file
audit did not locate a separate raw successful transfer receipt or HTTP response
archive. Current remote attempts failed. Before presenting these results as a
verified author-data comparison in a publication, retrieve the corresponding OSF
file/version independently and compare bytes/hash, or obtain confirmation and a
reference copy from the authors. No researcher has been contacted. Preserve the
existing files and results while that check is pending; do not replace missing
source data with invented values or reconstruct it from plotted means.

The comparison uses the authors' released per-sample TAALED CSV, not a table of
15 metrics printed in the main article. Earlier conversational wording blurred
these two sources. The numerical results are unchanged; their scope is clarified.

## Source chain

1. Baese-Berk et al. (2021), DOI https://doi.org/10.3389/fpsyg.2021.661415.
   The Data availability statement links OSF; Footnote 2 identifies the released
   data as exports from TAALED and TAALES.
2. Paper-linked repository:
   https://osf.io/dfhpu/?view_only=49d95d90424941da82217a239ab7450c
3. Author CSV download:
   https://osf.io/download/8jrpa/?view_only=49d95d90424941da82217a239ab7450c
4. Local unchanged file: dataset/public/lucid_published/TAALED_results.csv.
   94,436 bytes; 240 sample rows; 40 columns (including metadata).
   SHA-256: f4e9ad97e50d859311f04502a2c462826a783b064a5e4ce6e4282b058b947b63.
   This matches the hash in the saved OSF API listing at
   outputs/validation/lucid_validation/osf_listing.json, file GUID 8jrpa,
   OSF file ID 601468e6dd2225034e58b1e6. The saved listing dates the file to
   2021-01-29. Local source.json records the original acquisition URL/hash/size.
5. The downloaded author script LDLS_Analysis.R reads TAALED_results.csv at line11
   and references the same column identifiers in its models.

Fresh access on October16 returned HTTP500 from the CSV and R-script download
endpoints; the OSF API request timed out. Thus this recheck verifies the local file
against saved acquisition metadata, not against a newly downloaded remote copy.
See outputs/validation/lucid_validation/provenance_recheck_20261016.json.

## Exactly what we compared

The current retained comparison contains these15 CSV columns:

| CSV column | Meaning | Relation to main article |
|---|---|---|
| basic_ntokens | Total eligible tokens | Condition means in Table1 |
| basic_ntypes | Distinct processed lexical types | Additional released export field |
| basic_ncontent_tokens | Content-word tokens | Additional released export field |
| basic_ncontent_types | Distinct content-word types | Additional released export field |
| basic_nfunction_tokens | Function-word tokens | Additional released export field |
| basic_nfunction_types | Distinct function-word types | Additional released export field |
| lexical_density_tokens | Content tokens / all tokens | Additional released export field |
| lexical_density_types | Content types / all types | Additional released export field |
| simple_ttr_aw | Type-token ratio, all words | Figure1 |
| log_ttr_aw | Logarithmic TTR | Additional released export field |
| maas_ttr_aw | Maas TTR measure | Export field; author R script also analyzes it |
| mattr50_aw | Moving-average TTR, 50-word window | Figure2 |
| msttr50_aw | Mean segmental TTR, 50-word segments | Additional released export field |
| hdd42_aw | HD-D with sample size42 | Figure3 |
| mtld_original_aw | Original MTLD | Export field and author R script; NOT Figure4 variant |

Figure4 specifies a moving-average MTLD approach in both directions. The export
and author script contain mtld_ma_bi_aw as well as mtld_original_aw. We compared
only the original variant; we did not reproduce Figure4. There is no valid basis
to relabel original-MTLD agreement as agreement with that figure.

## Simple numerical cross-check

Arithmetic means of basic_ntokens from the unchanged CSV versus article Table1:

| Condition | CSV mean rounded to2 decimals | Article Table1 |
|---|---:|---:|
| No barrier | 662.78 | 662.78 |
| L2 | 1095.92 | 1095.92 |
| Babble | 756.75 | 756.75 |
| Vocoder | 785.22 | 785.21 |

The0.01 vocoder discrepancy is retained explicitly; its cause is not established.
The reference file was not changed to force agreement with the printed table.

## Claims supported by our analysis

Our analysis measures per-sample differences between our computed values and15
columns in the authors' released data. It scores180 rows and retains60 ambiguous
vocoder mappings as unavailable. It does not reproduce every figure, the TAALES
sophistication analyses, the mixed-model coefficients/p-values, or a published ML
accuracy benchmark. The percentage errors and engineering thresholds are ours,
not numbers or acceptance standards reported by the paper. The exact historical
cleaned input texts and complete original software configuration remain unavailable.
