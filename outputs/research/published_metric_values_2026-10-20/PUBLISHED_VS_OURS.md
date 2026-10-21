# Published results compared with our saved outputs

20 October 2026. These tables compare published values with existing saved pipeline outputs. No model inference, training, metric scoring or selection was run. Arithmetic summaries use the existing per-sample metric pairs.

## Matched samples: Baese-Berk et al. (2021), LUCID / TAALED

[Paper](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.661415/full) and [author data](https://osf.io/dfhpu/?view_only=49d95d90424941da82217a239ab7450c). The freshly retrieved author CSV matches all 2,700 scored reference cells, with matching sample, speaker and condition identifiers. Each condition has 60 samples from 20 speakers. All means below use those same samples.

The error column is the previously calculated mean participant absolute percentage error, averaged within sample/speaker. It is not the percentage difference between the two displayed means. Opposing sample errors can cancel in the overall mean.

| Condition | Metric | Published mean | Our mean | Existing mean participant APE % |
|---|---|---:|---:|---:|
| noBarrier | Word count | 662.78 | 669.30 | 16.60 |
| noBarrier | Different words | 205.75 | 207.55 | 12.05 |
| noBarrier | TTR | 0.3174 | 0.3172 | 7.56 |
| noBarrier | MATTR50 | 0.6975 | 0.6981 | 2.11 |
| noBarrier | HD-D42 | 0.7690 | 0.7694 | 2.12 |
| noBarrier | Original MTLD | 39.80 | 39.84 | 10.49 |
| babble | Word count | 756.75 | 762.63 | 0.76 |
| babble | Different words | 184.42 | 187.07 | 1.76 |
| babble | TTR | 0.2534 | 0.2551 | 1.31 |
| babble | MATTR50 | 0.6242 | 0.6254 | 0.44 |
| babble | HD-D42 | 0.7296 | 0.7315 | 0.37 |
| babble | Original MTLD | 28.71 | 28.83 | 2.49 |
| L2 | Word count | 1095.92 | 1115.12 | 1.87 |
| L2 | Different words | 230.75 | 232.47 | 1.33 |
| L2 | TTR | 0.2157 | 0.2134 | 1.57 |
| L2 | MATTR50 | 0.6508 | 0.6495 | 0.44 |
| L2 | HD-D42 | 0.7391 | 0.7383 | 0.25 |
| L2 | Original MTLD | 32.03 | 31.93 | 2.42 |

Interpretation: the closest sample-level agreement is in babble/L2, especially MATTR50 and HD-D42. No-barrier word-count, word-type and original-MTLD errors remain materially larger despite close aggregate means. This measures reproduction of author reference values; the paper does not provide an independently measured pipeline error against which to claim superior accuracy.

The vocoder comparison is unavailable because our saved outputs have no resolved matched samples; published values alone are not used to fill that gap. Original MTLD is an author-export field, not the moving-average MTLD plotted in the article's Figure 4. The article's vocoder word-count mean differs from the released CSV by 0.01.

## Two earlier LUCID papers: descriptive mean comparisons

The following put our available means next to the published means. Exact participant matching and counting/cleaning equivalence with these earlier papers are not established, so these differences are not presented as measurement errors or proof of superiority.

| Paper / condition | Published mean | Our available mean | Difference | Relative difference of means | Unit |
|---|---:|---:|---:|---:|---|
| Baker_Hazan_2011 / noBarrier | 613.00 | 669.30 | +56.30 | +9.18% | words per speaker per conversation |
| Hazan_etal_2015 / babble | 2157.00 | 2287.90 | +130.90 | +6.07% | words per speaker across three picture tasks |
| Hazan_etal_2015 / L2 | 3170.00 | 3345.35 | +175.35 | +5.53% | words per speaker across three picture tasks |

Sources: Baker & Hazan (2011), [DiapixUK paper, manuscript p. 13](https://discovery.ucl.ac.uk/1224322/1/Hazan_baker%20and%20Hazan%20BRM_revised_final.pdf); Hazan, Uther & Granlund (2015), [ICPhS paper, section 3.1](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2015/Papers/ICPHS0346.pdf). The 2015 comparisons sum all three existing samples for each speaker before averaging across speakers. The 2011 published quiet sample covers 40 speakers; our matched 2021 no-barrier subset has 20. These are different sample aggregations and should remain separate from the directly matched table above.

## OCSC and NYU–Emerson

Wagner et al. (2025) plots OCSC MLU and utterance counts and reports regression effects, but no verified exact raw group mean table. It would be invalid to compare its MLU age coefficient 0.321 with our raw MLU or percentage error as if they were the same quantity. Accordingly no numeric better/worse comparison is fabricated. NYU–Emerson has no verified published table of these language measures to compare.

## Reproducibility

- External reference: `sources/TAALED_results.csv`; byte provenance: `source_verification.json`.
- Existing pipeline metric values: `outputs/results/transcript_validation/lucid_metric_pairs.csv`.
- Existing participant-averaged errors: `outputs/results/transcript_validation/lucid_summary.csv`.
- Full precision comparisons: `published_vs_ours.csv`; identity/value verification: `comparison_verification.json`.
- Earlier published metric extraction and exact source locations: `PUBLISHED_VALUES.md`.
