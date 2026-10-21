"""Text-only, unreviewed utterance predictions with source provenance.

Never treats a speaker row as a verified utterance. No transcript is sent to
an external service. Rules are a research baseline, not a validated C-unit model.
"""
import re
import statistics
from openpyxl import load_workbook

METHOD = "chat_sentence_predictions_shared_clean_text_v4"
POLICY = {
    "name": "CLAN MLUw with Brown-listed filler exclusions; provisional text analysis",
    "segmentation": "https://talkbank.org/0info/manuals/CHAT.pdf#page=59",
    "automatic_boundary_rule": "Target-speaker turns split at spaCy sentence boundaries; dependency-parser coordinate-clause splits disabled after development-set error audit",
    "mlu": "https://talkbank.org/0info/manuals/CLAN.pdf#page=99",
    "fillers": "CLAN implementation point 4: uh/um; Brown rule 4: mm/oh",
    "repetition": "Brown rule 3 / CHAT 9.4: preserve separate utterances; unresolved internal repetitions flagged",
    "sampling": "Full recording; NOT Brown's 100-qualifying-utterance sample",
    "limitations": "Predicted boundaries and surface words only; no clinical certification, audio review or morphology",
}
FILLERS = {"uh", "um", "mm", "oh"}
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def read_rows(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(workbook["Transcript"].values)
        headers = [str(x).strip().lower() for x in next(rows)]
        for number, values in enumerate(rows, 2):
            row = dict(zip(headers, values))
            yield {"row": number, "speaker": str(row["speaker"]).strip(),
                   "text": str(row.get("transcript") or "").strip(),
                   "start": str(row.get("start time") or ""),
                   "end": str(row.get("end time") or "")}
    finally:
        workbook.close()


def target_turns(rows, speaker):
    """Merge target rows; allow a narrow predicted backchannel continuation.

    CHAT 9.10.2 supports continuation across backchannels. The text-only
    recognition below is an explicit heuristic, not a clinical classification.
    It only applies when preceding target speech has no final punctuation.
    """
    turn = []
    for row in rows:
        if row["speaker"].casefold() != speaker.casefold():
            backchannel = row['text'].strip().lower().rstrip('.!?') in {'mhm', 'mm-hmm', 'uh-huh', 'yeah'}
            if turn and backchannel and not re.search(r'[.!?]\s*$', turn[-1]['text']):
                continue
            if turn:
                yield turn
                turn = []
        elif row["text"]:
            turn.append(row)
    if turn:
        yield turn


def split_turn(text, nlp):
    """Split a target-speaker turn only at predicted sentence boundaries.

    An earlier version also split dependency-parser ``conj`` nodes.  On the
    corrected development transcripts those extra cuts included parser errors
    (for example, segments beginning with punctuation or noun phrases) and
    systematically inflated utterance counts.  Coordinate structure is still
    retained in the text and flagged for future human boundary review rather
    than treated as an automatic utterance boundary.
    """
    doc = nlp(text)
    cuts = {0: "turn_start", len(text): "turn_end"}
    for sent in doc.sents:
        cuts.setdefault(sent.start_char, "sentence_prediction")
    positions = sorted(cuts)
    for begin, end in zip(positions, positions[1:]):
        reason = cuts[begin]
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if begin < end:
            yield begin, end, text[begin:end], reason


def predict(rows, speaker, nlp):
    predictions = []
    for turn_id, turn in enumerate(target_turns(rows, speaker), 1):
        text, spans = "", []
        for row in turn:
            if text:
                text += " "
            start = len(text)
            text += row["text"]
            spans.append((start, len(text), row))
        for begin, end, body, reason in split_turn(text, nlp):
            sources = [r for a, b, r in spans if a < end and b > begin]
            predictions.append({"utterance_id": len(predictions) + 1,
                "turn_id": turn_id, "text": body, "turn_char_start": begin,
                "turn_char_end": end, "source_rows": [r["row"] for r in sources],
                "source_start": sources[0]["start"], "source_end": sources[-1]["end"],
                "timing_status": "source_segment_envelope_not_utterance_alignment",
                "boundary_reason": reason, "reviewed": False,
                "crosses_intervening_rows": any(b['row'] > a['row'] + 1 for a, b in zip(sources, sources[1:])),
                "scoring": score_utterance(body),
                "boundary_review_required": True})
    return predictions


def analysis_text(text):
    """One shared filler-free representation for exports and both lexical tools.

    Return empty for MLU-ineligible material. Never edit the raw source text.
    Surface punctuation is normalized, with final sentence/question type kept.
    """
    scoring = score_utterance(text)
    if not scoring['include_mlu']:
        return ''
    ending = '?' if text.rstrip().endswith('?') else '!' if text.rstrip().endswith('!') else '.'
    return ' '.join(scoring['counted_words']) + ending


def score_utterance(text):
    """Documented exclusions, with unresolved clinical choices left visible.

    This is a plain-ASR-text entry point, NOT a CHAT parser. It never guesses
    retrace codes or removes repeated words as though clinically classified.
    Explicitly coded CHAT must be processed with CLAN/the reviewed interface.
    """
    if any(c in text for c in "[]<>@&+*=|\\\x15") or "//" in text:
        raise ValueError("Coded/ambiguous source annotations require review; plain text expected")
    words = WORD.findall(text)
    flags = []
    lower = [w.casefold() for w in words]
    # A screening flag, not a rule for clinical inclusion/exclusion. Detect
    # repeated words and short repeated spans; do not conflate them with stutters.
    repeated = any(lower[i:i+width] == lower[i+width:i+2*width]
                   for width in range(1, min(5, len(lower)//2) + 1)
                   for i in range(len(lower)-2*width+1))
    if repeated:
        flags.append("repetition_or_emphasis_requires_audio")
    if any('-' in w for w in words):
        flags.append("hyphen_or_fragment_word_grouping_requires_review")
    if any("'" in w or '’' in w for w in words):
        flags.append("surface_contraction_count_not_morphology")
    if any(w in {"mm", "oh"} for w in lower):
        flags.append("Brown_listed_filler_assignment_requires_review")
    if re.search(r"\.\.|\b(?:I mean|sorry)\b", text, re.I):
        flags.append("possible_repair_or_abandonment_requires_audio")
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        flags.append("possible_multiword_name_requires_word_grouping_review")
    removed = [w for w in words if w.casefold() in FILLERS]
    counted = [w for w in words if w.casefold() not in FILLERS]
    reason = "unintelligibility_marker" if any(w in {"xxx", "yyy", "www"} for w in lower) else None
    if not counted and reason is None:
        reason = "no_eligible_words"
    return {"surface_words": words, "counted_words": [] if reason else counted,
            "excluded_fillers": removed, "include_mlu": reason is None,
            "exclusion_reason": reason, "word_count": len(counted) if reason is None else None,
            "review_flags": flags,
            "word_grouping_status": "surface_tokenization_not_reviewed_clinical_grouping"}


def word_mlu(utterances):
    lengths = []
    flagged = 0
    excluded = 0
    for utt in utterances:
        scoring = score_utterance(utt["text"])
        flagged += bool(scoring['review_flags'])
        if scoring['include_mlu']:
            lengths.append(scoring['word_count'])
        else:
            excluded += 1
    n = len(lengths)
    total_words = sum(lengths)
    return {"total_utterances": len(utterances), "mlu_nutterances": n,
            "total_words": total_words, "mlu_nwords": total_words,
            "mlu_words": total_words / n if n else None,
            "mlu_words_sd": statistics.pstdev(lengths) if n else None,
            "mlu_words_sample_sd": statistics.stdev(lengths) if n > 1 else 0.0,
            "mlu_morphemes": None, "mlu_status": "provisional_documented_exclusions_pending_review",
            "clinical_mlu_words": None, "flagged_utterances": flagged,
            "excluded_utterances": excluded, "policy": POLICY['name']}


def chat_text(utterances, source):
    """Narrow CHAT: no invented age, diagnosis, morphology, or media alignment.

    Word spelling retained; punctuation normalized. Full original text and
    source envelopes remain in the prediction JSON. Do not infer disfluency codes.
    """
    lines = ["@UTF8", "@Begin", "@Languages:\teng",
             "@Participants:\tPAR Target Participant",
             "@ID:\teng|speechpipeline|PAR|||||Participant|||",
             f"@Comment:\tSource: {source}",
             f"@Comment:\tAutomatic unreviewed predictions: {METHOD}.",
             "@Comment:\tText only; provisional boundaries and word grouping; no MOR review.",
             "@Comment:\tShared analysis text: uh um mm oh removed; ineligible units omitted.",
             "@Comment:\tInternal repetitions unresolved; no retrace codes invented."]
    for utt in utterances:
        text = utt["text"]
        if any(c in text for c in "[]<>@&+*=|\\\x15") or "//" in text:
            raise ValueError(f"Unsupported source annotation in {source}, utterance {utt['utterance_id']}")
        scoring = score_utterance(text)
        if not scoring['include_mlu']:
            continue
        words = scoring['counted_words']
        ending = "?" if text.rstrip().endswith("?") else "!" if text.rstrip().endswith("!") else "."
        lines.append("*PAR:\t" + " ".join(words) + " " + ending)
    return "\n".join(lines + ["@End", ""])
