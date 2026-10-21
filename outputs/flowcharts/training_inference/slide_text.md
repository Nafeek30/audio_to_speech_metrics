# Slide 1 — Our transcript pipeline

- Start with a transcript containing words and speaker labels.
- Keep the target speaker’s words and merge consecutive rows within speaker turns.
- Use MiniLM, grammatical rule suggestions, and a trained classifier to predict utterance endings.
- Calculate vocabulary measures and supported utterance measures using native CLAN.
- Save the predicted utterances, CHAT files, and result tables.

# Slide 2 — How training worked

- OCSC and NYU–Emerson human transcripts supplied the training examples.
- Human utterance breaks were hidden from the input and saved separately as correct answers.
- MiniLM provided word-context features; spaCy + hard-coded rules supplied a 0/1 suggestion at each gap.
- The classifier predicted CONTINUE or END, and training compared those predictions with the human answers.
- Training adjusted MiniLM, the rule embedding, and the classifier; spaCy and our rule code stayed fixed.
- The error calculation penalized wrong gap predictions and an incorrect expected boundary count.
- Different children were kept in training, validation, and test groups; validation helped choose the saved model.

# Slide 3 — How inference works

- Give the saved model new target-speaker text, without human boundary answers.
- The tokenizer converts words into token IDs, and MiniLM produces contextual features.
- spaCy + our hard-coded rules produce a 0/1 boundary suggestion, converted through a learned embedding.
- The classifier combines both inputs and predicts CONTINUE or END at each word gap.
- The resulting utterances feed the language-measure calculations; model weights do not change during inference.

# Optional one-sentence explanation

We fine-tuned MiniLM and trained a classifier to combine word context with grammatical boundary suggestions, using human-marked utterance endings as the training answers.

# Optional speaker notes

- The rule suggestion is an input, not the correct answer; the classifier can override it.
- Training loss combines gap-level classification error with a boundary-count penalty.
- The retained MiniLM checkpoint was selected from epoch 5. The later production decision margin was selected on validation participants.
- These diagrams describe the retained transcript workflow. They do not show a newly trained model or a new evaluation.
- Audio acquisition and audio-to-transcript processing are separate from the transcript workflow illustrated here.
