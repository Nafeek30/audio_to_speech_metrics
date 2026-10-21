# Why NYU is harder: measured differences and interpretation

These comparisons use the original **train and eval cohorts only**. They describe
input differences that inform development hypotheses. The current four-recipe CV
grid is registered separately; these calculations do not modify source data or cohort roles. Error rates are reported separately from these
pooled input statistics. This is an educated explanation, not a causal experiment.

| Input characteristic, pooled train + eval | NYU–Emerson | OCSC |
|---|---:|---:|
| Participants / task sections | 24 / 77 | 256 / 692 |
| Median target-child age in years, CHAT `@ID` | 4.58 | 7.22 |
| Source words | 6,592 | 283,546 |
| Median words per task section | 60 | 320 |
| Median words per observed target-speaker turn | 2 | 5 |
| Observed turns with at most 3 words | 70.07% | 40.56% |
| `xxx` / `yyy` / `www` per 1,000 source words | 79.49 | 5.73 |
| Native reference FREQ words absent from MLU word totals | 7.22% | 1.79% |

Within the training cohort, 11.80% of possible nonfinal NYU gaps are masked from
supervision, versus 1.08% for OCSC. The fixed label policy masks gaps adjacent to
unintelligibility tokens. NYU therefore supplies both fewer source words and a
smaller proportion of reliable supervised gaps.

**Less training information.** Original training contains 20 usable NYU children
and 211 OCSC children: 5,336 versus 229,378 source words. Corpus-balanced sampling prevents OCSC from dominating
optimization, but repeating the smaller NYU sample does not create more independent
children or more varied language examples. Five-fold CV shows how the fixed training
procedure behaves across held-out children; it does not increase this sample size.

**Different participant ages.** Target-child age is available for all24 NYU and
256 OCSC train/eval children. The median is4.58 versus7.22 years. The younger NYU
sample is another plausible reason that its language contexts differ from the
larger OCSC sample. Age, corpus, task and transcription practices vary together,
so this descriptive comparison cannot isolate an age effect or establish that
age causes a particular metric error. Age is not added as a model feature.

**Shorter, less informative contexts.** Most NYU turns contain no more than three
words. The model has fewer neighboring words to resolve whether speech continues,
ends, or forms a fragment. A single boundary or word-count difference can also cause
a large percentage change when the reference count or MLU is small. This makes
relative error harder to reduce even when many individual words are retained.

**More missing speech and eligibility differences.** NYU contains roughly fourteen
times as many unintelligibility tokens per source word. Missing speech hides language
cues that a text-only model cannot reconstruct. The larger native FREQ/MLU word-total
gap is consistent with stronger effects of utterance eligibility and original CHAT
coding. Our plain predictions preserve the words and do not invent human retrace,
error or unintelligibility scopes. Correct lexical counts therefore do not guarantee
identical eligible utterance counts or MLU. These statistics support the explanation;
they do not apportion the model error between individual annotation conventions.

**The difficulty is metric-specific.** In the preceding fixed-recipe five-fold study, NYU's
micro boundary F1 is 88.81%, versus OCSC's 87.22%, while mean participant eligible-count
error is 16.40% versus 9.16% and MLUw error is 10.57% versus 9.09%. Thus the results do
not support a blanket claim that every NYU boundary measure is worse. Boundary F1
and participant percentage errors weight different units, and native MLU also applies
eligibility conventions. The count/MLU weakness should be explained at that level.

**LUCID tests a different part of the pipeline.** LUCID agreement is measured against
the authors' published per-sample TAALED lexical results. Its word intervals do not
provide gold utterances or MLU. The successful historical dictionary/POS/pause
correction for babble and L2 supports the lexical calculation layer. It does not
establish that the boundary model generalizes to NYU. No-barrier lexical errors and
unresolved vocoder mappings remain reported. Likewise, OCSC is stronger relative
to NYU on several segmentation measures, but it still misses the declared targets.

Reproducible descriptive inputs and script:
`outputs/validation/cv_hyperparameter_v4/development_input_profile.json` and
`profile_development_inputs.py`. The separate CHAT-age summary is
`outputs/validation/cv_hyperparameter_v4/development_age_profile.json`.
Original participant assignments and all words remain
unchanged. The complete five-fold CV ranking selects the training recipe; original
eval selects the fresh final checkpoint and margin. These
explanations do not justify test-based tuning or dropping difficult recordings.
