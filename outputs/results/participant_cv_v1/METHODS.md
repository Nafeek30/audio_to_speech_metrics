# Participant-independent model development and evaluation

This describes the registered procedure. Completion and selected settings are recorded separately in `REPORT.md`, each job's `selection.json`, and `outputs/validation/participant_cv_v1/job_status.json`.

## Data and participant cohorts

The segmentation experiment uses OCSC and NYU–Emerson. Train, eval (validation), and test are disjoint at the participant level. Every task section from a participant retains the same original cohort assignment. LUCID is a separate lexical-comparison benchmark and does not supply segmentation-training examples.

| Corpus | Train participants / sections | Eval participants / sections | Test participants / sections |
|---|---:|---:|---:|
| OCSC | 211 / 567 | 45 / 125 | 45 / 111 |
| NYU–Emerson | 20 / 63 | 4 / 14 | 4 / 15 |

These are usable-data counts. Two additional NYU participants retain their original training assignments but provide no eligible prepared boundary-scoring sections. Two OCSC source files contain no selected discourse task. Their exclusions and all source-file hashes remain in the cohort manifest; they are not treated as zero-error observations.

## Input preparation

The target child is identified using the transcript speaker labels. Non-speech event descriptions are removed, while spoken repetitions, fillers and fragments remain. Human punctuation, timestamps, morphology, individual target-row endings and reference utterance boundaries are withheld from the model. Consecutive target-speaker rows are merged; observed changes to another speaking participant separate inference contexts. Source words are preserved exactly apart from the documented tokenizer normalization used internally by the encoder.

Reference utterances use original human CHAT, including explicit continuation handling. Human references and prediction exports are distinct. Predicted CHAT contains the retained words and predicted boundaries, with unambiguous `uh`/`um` pause coding, without invented retrace, grammatical, morphological or maze annotations.

## Custom model

The encoder initializes from Microsoft's original `MiniLM-L12-H384-uncased` checkpoint at revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`. This is pretrained initialization followed by task-specific fine-tuning, not training a language encoder from scratch.

For each word gap, the classifier concatenates the contextual representations at the left and right word boundaries with a learned 16-dimensional embedding of an automatically derived linguistic rule feature. A 128-unit hidden layer, GELU activation, dropout 0.15 and a two-class output predict boundary versus no boundary. Encoder, feature embedding and classifier weights are optimized jointly.

Word-based overlapping windows contain up to 256 subword tokens, with stride 128. Training assigns each valid gap to one central window. Inference combines overlapping emission differences with center weights. The geometry depends on source words, not reference cuts.

Training uses cross entropy plus a fixed-coefficient relative boundary-count penalty. Training windows have equal expected sampling mass per corpus. One seeded word-position crop per training paragraph is added; crop-final gaps are masked and every crop must match its own fit-participant parent. Five percent of non-special training subtokens are replaced by UNK. No augmentation is generated from an inner-validation, outer-held-out, original-eval or original-test participant for that model.

The fixed optimization settings are AdamW, encoder learning rate 2e-5, head/embedding learning rate 1e-4, weight decay 0.01, batch size 8, two-batch gradient accumulation, 10% warmup, gradient clipping at 1.0, at most six epochs, and patience two. No parameter averaging is used in this study.

## Five-fold cross-validation with inner participant holdouts

Only the original training cohort (231 participants, 630 sections) enters cross-validation. Participant IDs are shuffled using the registered seed and distributed across five outer folds separately within each corpus. Each participant receives exactly one outer-held-out prediction. All of that participant's sections are held out together.

Within the other four outer folds, 20% of participants per corpus form an inner validation group. The remaining participants fit the weights. Thus each outer model fits 147–148 children, selects on 37 different inner-validation children, and is evaluated on 46–47 different outer-held-out children. The original eval and test cohorts remain outside all five CV jobs.

Each fold starts independently from the original pretrained encoder, with a new task head and registered seed. No fold starts from the preceding fold or the previous fine-tuned production model. Initial encoder and head hashes, actual participant roles, augmentation provenance, training history and selected weights are retained.

The epoch is selected by the lowest inner-validation objective at zero margin. After that checkpoint is selected and reloaded, the seven registered margins 0, 0.25, 0.5, 0.75, 1, 1.5 and 2 are scored on inner validation only. One global margin minimizes the same objective. Ties retain the earlier epoch or the first listed margin. The chosen epoch and margin are frozen before outer evaluation. Outer results never select a model, margin or production fold.

This is five-fold outer cross-validation with an inner holdout split, not a 5-by-5 inner grid of cross-validation fits.

## Separate final model

A sixth model starts from the original pretrained encoder and a new task head. It fits all original training participants. The original eval cohort selects its epoch and global margin using the same fixed procedure. The selected weights and settings are frozen before loading the original test cohort. The final model is not chosen by comparing outer-fold or test scores.

## Metrics and reporting

Native CLAN MLU (`-t%mor`) and FREQ (`+k +b50`) score original human CHAT and predicted CHAT under the same surface-metric commands. Eligible MLU utterance count, MLUw and length SD are distinct from vocabulary measures and from the total number of verbal units. Native displayed precision is retained for SD/MATTR. Unsupported annotation-dependent measures remain unavailable.

For each measure, absolute percentage error is computed within section, averaged across sections within participant, then equally across participants within corpus. The selection objective equally averages the eligible-count, MLUw and SD errors across both corpora. This prevents OCSC sample size from dominating model selection. Zero reference denominators and unavailable metrics remain missing rather than becoming zero errors.

Out-of-fold predictions are pooled once per original training section. Reports include per-corpus participant-level error, individual fold values, fold mean and sample SD, boundary precision/recall/F1, and exact-span F1. Participant-cluster bootstrap intervals resample participants with all their sections together, conditional on the saved predictions. Fresh final-model test results are presented separately from CV results.

The data/metric audit checks disjoint participant roles, original cohort preservation, fit-only crops, fresh model initialization, inner-only selection, freeze-before-evaluation chronology, exactly-once outer coverage, unchanged word streams and human reference values, native output hashes, reproduced metric summaries and reproduced boundary/span counts.

## Scope

These are participant-independent transcript-based segmentation evaluations. They do not evaluate automatic speech recognition or speaker diarization. LUCID lexical agreement is assessed separately against the authors' published TAALED table; it does not establish utterance segmentation accuracy. Earlier experimental records remain preserved separately from these fresh models and cross-validation outputs.

## Prospective model-improvement extension

Before fresh final-test scoring, the user requested using CV to improve accuracy.
A separate immutable protocol is registered under
`outputs/validation/participant_improvement_v2/protocol.json`. The parent five-fold
training recipe, splits, seeds, loss, inner selection and OOF scoring are unchanged.
The parent final single-model test is held explicitly. Its freshly trained full-train
model is a component, rather than an automatically accepted production replacement.

On original eval only, compare (1) the full-train model, (2) the equal-probability
ensemble of five fold models, and (3) the equal-probability ensemble of all six.
For each, use the same seven previously fixed nonnegative boundary margins. Mean
boundary probabilities are converted to log odds before applying a margin;
singleton predictions preserve original logits exactly. All candidates share input
speaker-turn contexts, features, unchanged word streams and native CLAN metrics.
Select minimum corpus-balanced participant mean APE across eligible count, MLUw
and length SD; ties retain listed candidate and margin order. Freeze selected
candidate, margin and component hashes before opening original test. Test scores
never choose an ensemble, margin, member weight or deployment.

The five fold models have different freshly initialized heads, so their parameter
weights must not be averaged. Probability averaging is appropriate here. Every
member excludes original eval/test participants from gradients. The original-train
pool must not be scored by the ensemble and called OOF: some members trained on
those children. The OOF report uses only the dedicated outer holdout model.

A separate deployment policy was registered before candidate evaluation: retain
the prior workflow unless the new candidate improves its identical original-eval
objective (11.35342133442171%). Final test outcomes do not trigger deployment.
Runtime parity must pass all 139 original-eval sections before a new workflow can
be activated; both corpus CLI paths must reproduce saved native counts/utterances.

## Pretest calibration refinement (v3)

The fresh full-training model selected epoch 3 and margin 0 on original eval,
objective 11.8071%, versus 19.8952% at margin +0.25. The optimum lay at the lower
endpoint of the original nonnegative grid. Before any fresh test was opened, a
separate v3 protocol registered five additional margins (-1, -0.5, -0.25, -0.125,
+0.125), giving 12 margins for each of the same three model candidates (36 total).
This checks both directions for newly trained score calibration. All six model
weights, original cohorts, original five-fold settings and OOF results are unchanged.
The ten completed v2 development comparisons and all six source/model-hashed emission
caches are retained and reused. A selected cached result must reproduce its native
objective/summary before freezing. No further refinement follows final test scoring.

New outputs/evidence are `participant_improvement_v3`; source is
`speech_pipeline.research.participant_improvement_refined`. The same prior-production
eval guard remains in effect, and final test still evaluates only the frozen winner.
