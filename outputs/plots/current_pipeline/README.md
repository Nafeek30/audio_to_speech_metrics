# Current pipeline plots

Four slide-ready PNGs, white backgrounds, 3200 × 1800 pixels (200 DPI).
HD-D42 is omitted from all plots. Dataset names are written in full.

1. Child-corpus test vocabulary errors: TNW, NDW, TTR, MATTR-50.
2. Child-corpus test utterance errors: total utterances, MLUw, and length SD.
3. LUCID combined lexical errors: total/different words, TTR, MATTR-50.
   Includes all 180 scored samples across babble, English-learner partner and
   no-barrier conditions. APE is averaged within each speaker across their available
   samples, then equally over 40 speakers. This descriptive aggregate is newly
   calculated from saved errors, not a new evaluation run or a simple mean of
   the three condition means. Unmatched vocoder samples remain unscored. The
   author-reference source remains pending independent confirmation.
4. Five-fold CV tuning: baseline versus lower count-loss weight; separate from
   existing production-test results. The baseline remains best combined among
   completed configurations; this figure shows the tradeoff, not a new final model.

Word-count error is absolute percentage error in the count, not ASR word error rate.
Saved errors average within sample, participant and corpus as documented in the
source reports. Definitions/values/source hashes are in plot_data.json.
The generator is create_plots.py. No research source file or model was modified.
All four rendered PNGs were visually reviewed and verified for dimensions.

Figure 3 title and subtitle simplified at user request: the title no longer says
provisional and the second subtitle is removed. Source confirmation status and
coverage remain documented here and in plot_data.json; numerical results are unchanged.

Main titles removed from all four PNGs at user request. Subtitles, dataset labels,
axes, legends and numerical values remain. All four updated PNGs visually checked.

Figure 2 subtitle removed at user request; chart values, axes and legend retained.

Latest Figure 2 subtitle contains dataset-specific test participant counts only:
Ohio Child Speech Corpus, 45; New York University and Emerson College corpus, 4.
No retained-pipeline wording or section counts.
