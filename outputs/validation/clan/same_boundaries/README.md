# CLAN comparison with the same automatic utterance rules

There are 37 recording files in each input folder:

- **corrected_auto**: corrected transcripts segmented with the current pipeline's utterance code.
- **uncorrected_auto**: the matching current pipeline files, copied from the existing export. Paired files have identical filenames in their separate folders.

## What the previous comparison used

The uncorrected recordings used `automatic_utterances.py` (now `speech_pipeline/utterances.py`), method `conversation_clause_predictions_v5`. The corrected reference used one utterance per nonempty supplied text line. CLAN counted the exported eligible utterances; it did not determine their grammatical boundaries. TAALED received the cleaned utterance text for vocabulary measures and did not calculate utterance boundaries or MLU.

## What these files use

Both sides now use the same `speech_pipeline.utterances.split_turn` function and the same spaCy model version. No boundary rules were changed for this export.

- Join consecutive target-speaker segments into turns. Apply the existing narrow continuation rule across a brief intervening backchannel.
- Use the parser's sentence predictions, with guards against splitting leading fillers or repetitions.
- Keep dependent clauses with their main clause.
- Split a coordinated clause when the code identifies its own subject and predicate, with a qualifying conjunction. Keep coordinated phrases together and avoid cuts inside subordinate clauses.
- Apply the existing separate check for a subject and predicate after “but.”
- Keep short responses, repetitions, fillers, and abandoned speech in the segmented text.

Corrected files lack interlocutor labels and timestamps. The export therefore maps original speaker-turn breaks onto the corrected word stream using minimum-edit word alignment. Between uncertain exact-word anchors, it interpolates the turn break and records a review flag. It does not copy the uncorrected predicted utterance boundaries. Corrected line breaks are treated as whitespace; corrected wording and punctuation are retained. Boundary locations may differ as the wording and punctuation differ.

Of 1,350 source turn breaks, 379 require review because their surrounding words did not map exactly. These are approximate turn mappings, not audio-verified boundaries. The `utterances` folder preserves every corrected word and punctuation mark, apart from whitespace normalization, and records the mapping and segmentation reasons.

After segmentation, the MLU export removes the existing fixed filler list: `uh`, `um`, `uhm`, `hmm`, `hm`, `mm`, `mhm`, `huh`, `oh`, and spelling variants. It omits filler-only units and units containing `xxx`, `yyy`, or `www` from MLU. Repetitions remain. Each `*PAR:` line is one MLU-eligible utterance. Full utterance text with fillers remains in the JSON files; `manifest.csv` reports total and eligible utterance counts separately. Corrected eligible word counts are unchanged from the earlier corrected export in all 37 recordings.

This comparison examines the effect of corrected transcript text under one automatic boundary method. It is separate from evaluating the method against independently reviewed utterance boundaries. The paper and poster use these corrected references. The earlier source-line reference folder is retained for provenance.

## Run in CLAN

1. Open CLAN's **Commands** window (Command-D on a Mac if it is hidden).
2. Click **working**, choose **corrected_auto**, and select the current directory. Set the **output** directory to that same folder if it points elsewhere.
3. Run the format check:

   ```text
   check *.cha
   ```

4. Run word-based MLU:

   ```text
   mlu +t*PAR -t%mor +f *.cha
   ```

5. Repeat steps 2–4 for **uncorrected_auto**, keeping its output in that folder.
6. Keep the **37 `.mlu.cex` files per folder** with their original names. Return both folders together, including the `.cha` inputs, `.mlu.cex` outputs, and `manifest.csv`. Save any CHECK messages too, especially if you edit a flagged passage.

`+t*PAR` selects the target participant. `-t%mor` uses words on the speaker lines; no morphology tier is supplied. `+f` saves a separate output for each recording. MLUw is counted words divided by eligible utterances. The output also reports utterance count, word count, and utterance-length standard deviation. Run the command without `+u` so recordings remain separate.

## Checks already completed

All 37 returned corrected-transcript `.mlu.cex` outputs were imported. Their headers identify CLAN 31-Aug-2026 and the command `mlu +t*PAR -t%mor +f`. Input hashes match the manifest. Returned counts match the calculator exactly, and MLUw and standard deviations agree to three displayed decimal places. TAALED was rerun on the same corrected utterances. Results are in the existing `outputs/validation/method_comparison` and `outputs/results/paper` folders. Both input folders retain their 37 saved `.mlu.cex` outputs.

The official CLAN MLU program was run locally on all 74 files. Word and eligible utterance counts match the pipeline calculator exactly. Means and population standard deviations match to CLAN's displayed precision. Twenty boundary/scoring/export tests passed. The uncorrected speaker lines match the saved current analysis exactly.

CHAT CHECK passes in 71 files. Three P9 files contain numeric spellings in game names or years. `number_review.csv` lists their exact passages and physical file line numbers. MLU runs on those files, but numeric spellings require recording review for full CHAT validation. Write them as actually spoken if you review them, keep the edited inputs with their outputs, and rerun CHECK and MLU. The export preserves those words rather than guessing their pronunciation.

## Files for comparison and reproduction

- `manifest.csv`: paired filenames, predicted metrics, turn-mapping flags, and input hashes.
- `validation.csv`: local CLAN calculation and format-check status.
- `provenance.json`: method, code hashes, and parser versions.
- `utterances/*.json`: complete corrected utterances with fillers, source-turn mapping, boundary reasons, and expected metrics for both sides.

Regenerate from the project root using:

```sh
.venv/bin/python -m speech_pipeline.clan
```

The returned CLAN results can be compared in two ways: corrected versus uncorrected under the same boundary code, and CLAN versus our calculator on identical input. The second comparison checks calculations; it does not establish independent boundary accuracy.

Command and interface reference: [official CLAN manual](https://talkbank.org/0info/manuals/CLAN.html), sections 3.1 and 7.21.
