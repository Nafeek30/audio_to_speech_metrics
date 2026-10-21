# Adjacent slide text

## 1. Transcript word error rate
- Compared automatic transcripts with corrected transcripts: 37 recordings, 10 participants.
- Participant mean WER ranged from 4.15% to 27.04%.
- WER = (substitutions + deletions + insertions) / corrected word count. Lower is better.

## 2. Agreement with corrected TAALED and CLAN
- Applied the same automatic utterance rules to both transcript versions.
- CLAN MLUw error decreased from 15.27% to 4.68%; eligible utterance-count error decreased from 21.82% to 5.46%.
- TAALED lexical errors remained low: MATTR-50 error was 1.11%.

Presenter note: Previous/current here compares source-line versus automatic segmentation of the corrected reference, while retaining the current raw-transcript pipeline. This is a comparison-method change, not an independent boundary-accuracy improvement. Corrected transcripts lack speaker tiers; original turns were mapped by word alignment, with 379 of 1,350 turn breaks flagged for review. MAPE averages recording percentage errors within participants, then weights participants equally.

## 3. Agreement with labeled utterance samples
- Compared previous and revised pipeline rules with human labels: 14 recordings, 10 participants.
- Utterance-count error decreased from 11.14% to 7.96%; MLUw error decreased from 15.09% to 9.12%.
- Lexical errors stayed small; utterance-length SD still had 25.09% error.

Presenter note: The colors identify TAALED-style lexical and CLAN-style utterance measures; the reference is human labels. Count here includes all utterances, unlike the MLU-eligible count in Figure 2. Same participant-balanced MAPE definition as Figure 2.

## 4. Pipeline vs human utterance boundaries
- Jade: agreement 99.34% → 99.56%; Cohen’s κ 0.950 → 0.968.
- Yamen: agreement 94.02% → 94.82%; Cohen’s κ 0.853 → 0.874.
- Pooled: agreement 98.38% → 98.70%; Cohen’s κ 0.917 → 0.935.
- Each rater supplied two matched recordings. These are development results, not held-out validation.

Presenter note: Agreement counts matching boundary/no-boundary decisions at exactly aligned adjacent word gaps. Cohen’s κ adjusts for agreement expected by chance. Both include no-boundary matches; unmatched gaps are excluded. These are pipeline-versus-rater comparisons, not Jade-versus-Yamen agreement.

Comparator clarification: “Previous pipeline” and “Revised pipeline” are two automatic methods compared with the same rater. Jade’s pooled pipeline κ values (0.950 and 0.968) are not human interrater reliability. Rechecking the primary human worksheets against Jade, using the filler-excluding human comparison in speech_pipeline/research/human_interrater.py, gives κ = 0.749120 for P2 and 0.864632 for P9. These human-versus-human scores cannot be attributed to pipeline improvements. The saved outputs/results/paper/interrater_results.csv currently contains pipeline-versus-rater counts, despite its filename; do not cite it as human-versus-human reliability without correction.

# How we fixed utterance processing

- **Use grammar to split independent clauses.** Previously, sentence-only splitting could merge two complete clauses. The revised spaCy dependency rules require a conjunction, a predicate, and an explicit subject in the second clause. Example: “I went home and I made dinner.” → “I went home” | “and I made dinner.”
- **Protect dependent clauses and phrases.** “I went home because I was tired.” stays one utterance. “I went home and made dinner.” also stays one because the second verb shares the subject.
- **Keep fillers and repetitions during segmentation.** “Um, I I went home.” stays together; the rules suppress false parser boundaries after leading fillers or repeated words. Repetitions remain in the word count.
- **Separate segmentation from scoring.** “Um.” remains an utterance, but is excluded from MLU. For “Um.” | “I went home.” | “I made dinner.”: total utterances = 3; eligible utterances = 2; eligible words = 6; MLUw = 6/2 = 3 words.
- **Read human labels correctly.** A cell containing “It shrunk. Kind of. Yeah.” now becomes three labeled utterances. The parser handles periods even before lowercase text, while protecting abbreviations such as “Dr.” and hesitation ellipses.
- **Use the same boundary and scoring rules for the software comparison.** Corrected line breaks previously defined the CLAN reference utterances. Both transcript versions now pass through the same automatic splitter and filler policy. CLAN calculates MLU from the supplied boundaries; TAALED calculates lexical measures.

Verification: On identical exported input, CLAN matched the pipeline's eligible word and utterance counts exactly across all 74 files; means and SDs matched CLAN's displayed precision. This verifies calculation consistency, not independent segmentation accuracy.
