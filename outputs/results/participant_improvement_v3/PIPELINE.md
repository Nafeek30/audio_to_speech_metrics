# Pipeline and current model-improvement work

The scope is **transcripts → predicted utterances → language metrics**.
Audio transcription, diarization and the original ten-participant study are deferred.
The report and live status files identify which registered training stages have finished.

```mermaid
flowchart TD
    A[Target-speaker transcript and observed speaker labels] --> B[Retain spoken words; merge adjacent target rows into observed turns]
    B --> C[Original spaCy preprocessing and linguistic rule features]
    C --> D[Selected trained boundary model or probability ensemble]
    D --> E[Apply eval-selected boundary margin]
    E --> F[Predicted utterances; same source words]
    F --> G[Faithful predicted CHAT export]
    G --> H[Official CLAN MLU and FREQ]
    H --> I[Eligible utterance count, MLU in words, length SD, vocabulary measures]
    B --> J[Fixed pause cleaning and historical TAALED compatibility layer]
    J --> K[Lexical counts, density and diversity]
    K --> L[LUCID comparison against authors' published per-sample TAALED table]
    I --> M[NYU and OCSC comparison against original human CHAT scored with CLAN]
```

At inference, source utterance row endings, human reference cuts and clinical annotation
scopes are withheld from model inputs. During supervised training, reference cuts from
fitting participants supply the target labels. Observed speaker changes define inference contexts.
The prediction export does not invent morphology, retrace, maze or error annotations.
LUCID has lexical reference values; it does not provide gold utterances or MLU.

```mermaid
flowchart TD
    T[Original TRAIN: 231 usable participants / 630 sections] --> CV[Five participant-grouped folds]
    CV --> FIT[Each fold: fit participants, separate inner-validation participants]
    FIT --> FIVE[Five fresh fine-tuned models; inner-only epoch and margin selection]
    FIVE --> OOF[Each model scores only its outer-held-out participants]
    OOF --> R[OOF report: exactly one prediction per training section]
    T --> FULL[Sixth fresh model fitted on all original TRAIN]
    V[Original EVAL: 49 participants / 139 sections] --> EP[Select full-training model epoch]
    FULL --> EP
    EP --> CAND[Compare full-training single model, five-model ensemble, six-model ensemble]
    FIVE --> CAND
    V --> CAND
    CAND --> SEL[Eval-only selection across three candidates and twelve margins]
    SEL --> FREEZE[Freeze candidate, margin and component hashes]
    FREEZE --> TEST[Apply only the frozen choice to original TEST]
    TE[Original TEST: 49 participants / 126 sections] --> TEST
    TEST --> OUT[Final test metrics, native outputs and integrity audit]
    FREEZE --> GUARD[Deploy only if eval objective improves previous production; verify runtime parity]
```

All original train/eval/test assignments stay fixed and participant-independent.
Every section/crop from a child keeps that child's role for the relevant model.
Original eval/test participants never fit any member's weights. Test results do not
choose a candidate, margin, ensemble weight or deployment. The ensemble's original
training-pool score would not be OOF; only the dedicated held-out fold predictions
are used for that report.

| Component | What is supplied externally | What is ours |
|---|---|---|
| Language preprocessing | Existing spaCy English model | Input policy, word conservation and rule-feature extraction |
| Boundary encoder | Original pretrained Microsoft MiniLM | Task-specific fine-tuning on original training participants |
| Boundary classifier | None | Learned rule embedding and gap classifier; custom training/count loss |
| Model improvement | Standard probability-averaging principle | Registered grouped CV, fresh models, eval selection and runtime routing |
| Lexical calculation | Historical TAALED dictionary/category conventions | Compatibility implementation and fixed pause handling |
| CHAT calculation | Official native CLAN MLU/FREQ executables | Faithful export, cohort/reference alignment, error calculation and audit |

The segmentation model is custom task training built on a pretrained encoder.
It is not a language encoder trained from scratch, and it is not merely an unchanged
pretrained model. Lexical compatibility and native CLAN calculations do not learn
weights from LUCID, eval or test data.
