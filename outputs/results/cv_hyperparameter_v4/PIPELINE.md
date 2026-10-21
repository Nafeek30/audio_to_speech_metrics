# Current pipeline and completed comparisons

```mermaid
flowchart LR
 A[Target-speaker transcript] --> B[Observed turns and fixed word preparation]
 B --> C[Fixed spaCy rules and contextual features]
 C --> D[Retained custom MiniLM boundary model]
 D --> E[Margin 0.25 and predicted utterances]
 E --> F[Faithful CHAT export]
 F --> G[Native CLAN MLU and FREQ]
 B --> H[Historical TAALED lexical compatibility]
 H --> I[Lexical counts and diversity]
 G --> J[NYU and OCSC human CHAT references]
 I --> K[LUCID published TAALED values]
```

The boundary model uses an off-the-shelf pretrained MiniLM encoder plus our custom
learned rule embedding and gap classifier. Task training updates the encoder and
head jointly. SpaCy and native CLAN are external components. Original human
utterance endings stay outside inference inputs. All words remain preserved.

```mermaid
flowchart TD
 T[Original train: 231 usable children] --> R[Three completed training configurations]
 R --> CV[Same five participant-grouped folds per configuration]
 CV --> FIT[Fitting children supply gradients and crops]
 CV --> IN[Different inner children select epoch and margin]
 FIT --> M[Fresh original encoder and task head]
 M --> SEL[Freeze fold model and margin]
 IN --> SEL
 SEL --> HOLD[Score different heldout training children]
 HOLD --> POOL[Pool all five folds per configuration]
 POOL --> B[Baseline best combined CV score]
 POOL --> O[Lower count loss best OCSC CV mean]
 T --> U[Fourth setting interrupted in fold 1]
 U --> X[Preserve checkpoint and exclude whole setting]
 B --> STOP[User requested stopping]
 O --> STOP
 STOP --> KEEP[Retain previous production workflow]
 EV[Original eval] --> OLD[Earlier workflow and margin selection]
 OLD --> TEST[Existing original-test evaluation]
```

All15 completed assessments passed role, crop, initialization, frozen-weight,
word-preservation and native-metric checks. No original eval/test child entered
CV recipe comparison. The current search stopped before a new final fit or test.
CV tuning values and existing production-test results describe different fitted
models and evaluation roles. No new model was activated.

LUCID remains a separate lexical benchmark. Its word intervals do not provide gold
utterances or MLU. Agreement reproduces published lexical statistics and does not
establish superiority over a published trained boundary model. Babble/L2 results
are strong; noBarrier and unresolved vocoder remain qualified.
