# CLAN reference files

`same_boundaries/` is the automatic-boundary comparison used by the current saved results. It contains `corrected_auto/` and `uncorrected_auto/`, each with 37 CHAT inputs and 37 official `.mlu.cex` outputs. Its README, manifest, validation records, and utterance audits explain the procedure.

`source_lines/` preserves the earlier corrected reference with one supplied transcript line per utterance, plus its manifest and number-review record. This is a different boundary procedure, retained as supporting evidence.

In CLAN's Commands window, set the working and output directories to one input folder and run:

```text
check *.cha
mlu +t*PAR -t%mor +f *.cha
```

Repeat for the matching input folder. Keep each `.cha` file with its original `.mlu.cex` result. These files calculate word-based MLU; no morphology tier is supplied. CLAN counts the exported utterances and does not determine their grammatical boundaries.

The automatic export retains fillers and repetitions while segmenting. It removes the study's fixed filler list for word metrics and omits filler-only or unintelligible utterances from the MLU denominator. Complete utterance text and scoring decisions remain in the saved JSON reports.

The saved official runs match calculator word/utterance counts exactly and means/standard deviations to CLAN's displayed precision. Three P9 inputs contain numeric spellings requiring recording review for full CHAT validation; see `same_boundaries/number_review.csv`. Do not guess how the numbers were pronounced.
