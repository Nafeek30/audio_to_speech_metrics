# Alternative publication route: matched baselines and reference validation

Saved 20 October 2026 at the user's request. This is an alternative to locating existing published measurement-error benchmarks on new public datasets; it is not a completed experiment or an acceptance guarantee.

## Proposed contribution

Evaluate automatic utterance segmentation of supplied speaker transcripts, its effect on native language measurements, and the benefit, if demonstrated, of combining contextual word features with grammatical boundary suggestions. Distinguish calculator fidelity, segmentation accuracy and end-to-end audio accuracy.

## Evidence a methods reviewer would expect

1. On identical word inputs and available speaker-turn context, compare spaCy alone, spaCy plus rules, a MiniLM model trained without the rule feature, and the combined model. Use identical references, scoring and participant separation. Disabling a feature only at inference is not equivalent to a properly trained ablation.
2. Report boundary precision/recall/F1 alongside utterance-count, MLUw and utterance-length-SD errors. Count and MLU are mathematically related; errors can cancel, so neither downstream scores nor the number of reported metrics substitutes for boundary assessment.
3. Independently check a predefined representative reference subset with blinded trained annotators. Report human disagreements and adjudication. Independently spot-check counts and MLU calculations.
4. Report original-unit MAE, percentage errors, signed bias, large failures, agreement plots and participant-aware uncertainty. Do not convert percentage measurement errors into classification accuracy or equate correlation with agreement.
5. Define acceptable error for the intended use. Current 5%/10% goals are engineering targets, not established clinical thresholds.
6. Preserve NYU's small test size, mixed performance, historical benchmark exposure, incomplete stopped CV, LUCID coverage and metric-variant qualifications. Lexical fidelity on preserved words does not demonstrate a boundary-model improvement.

## Existing evidence and limits

The original transcript study remains stopped. No new training or production activation is authorized by saving this plan. Existing production-test count/MLUw/SD errors are OCSC 10.36/8.70/14.24% and NYU 29.98/16.57/13.70%. Those results cannot be mixed with selected CV tuning results. Current matched-input Batchalign evidence is historical and has its own scope and version limits.

## Priority if this alternative is chosen

Complete a matched baseline table, quantify uncertainty and perform an independent annotation audit. Claim improvement only if the measured comparison supports it. A useful validation/benchmark contribution need not win every endpoint.

Methodological references: [Bland & Altman, 1986](https://www-users.york.ac.uk/~mb55/meas/ba.pdf); [Gorman et al., 2015](https://aclanthology.org/W15-1213/).
