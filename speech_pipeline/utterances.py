"""Text-only, unreviewed utterance predictions with source provenance.

Never treats a speaker row as a verified utterance. No transcript is sent to
an external service. Rules are a research baseline, not a validated C-unit model.
"""
import re
from collections import defaultdict
import numpy as np
import statistics
from openpyxl import load_workbook

METHOD = "conversation_clause_predictions_v5"
POLICY = {
    "name": "Utterances retain fillers and repetitions; word metrics exclude listed fillers",
    "segmentation": "https://talkbank.org/0info/manuals/CHAT.pdf#page=59",
    "automatic_boundary_rule": "Sentence boundaries with guarded coordinate clauses requiring their own subject; subordinate clauses and coordinated phrases stay attached",
    "mlu": "https://talkbank.org/0info/manuals/CLAN.pdf#page=99",
    "fillers": "uh, um, uhm, hmm, hm, mm, mhm, huh, oh and spelling variants removed only for word metrics; all retained for total utterances",
    "repetition": "Retain all transcript repetitions; do not infer or remove stutters or retraces",
    "sampling": "Full recording; NOT Brown's 100-qualifying-utterance sample",
    "limitations": "Predicted boundaries and surface words only; no clinical certification, audio review or morphology",
}
FILLERS = {"uh", "um", "uhm", "hmm", "hm", "mm", "mhm", "huh", "oh"}
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def is_filler(word):
    return word.casefold() in FILLERS or bool(re.fullmatch(r"u+h+|u+m+|u+h+m+|h+m+|m+h+m+|m{2,}|h+u+h+", word.casefold()))


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
    """Keep dependent clauses/phrases; split a conjoined independent clause.

    A parser conj alone is insufficient: require a predicate, an explicit
    subject and a conjunction. Avoid cutting inside subordinate/relative
    clauses. Preserve every source character apart from surrounding whitespace.
    """
    doc = nlp(text)
    cuts = {0: "turn_start", len(text): "turn_end"}
    previous = None
    for sent in doc.sents:
        # Keep grammatical unpunctuated boundaries, but suppress parser cuts
        # inside repetitions and leading fillers ("I I I", "Um I went").
        terminal = bool(re.search(r'[.!?][\"\'’”)]*\s*$', text[:sent.start_char]))
        prev_words = [t.lower_ for t in previous if t.is_alpha] if previous is not None else []
        next_words = [t.lower_ for t in sent if t.is_alpha]
        false_fragment = previous is not None and not terminal and (
            all(is_filler(w) for w in prev_words) or
            len(set(prev_words)) == 1 and next_words and prev_words[0] == next_words[0])
        dependent_fragment = any(c.dep_ == 'mark' and c.lower_ in {'because', 'although', 'if', 'when', 'while', 'since'}
                                 for c in sent.root.children)
        if not false_fragment and not dependent_fragment:
            cuts.setdefault(sent.start_char, "sentence_prediction")
        previous = sent
        for predicate in sent:
            if predicate.dep_ != 'conj' or predicate.pos_ not in {'VERB', 'AUX', 'ADJ'}:
                continue
            children = list(predicate.children)
            subjects = [t for t in children if t.dep_ in {'nsubj', 'nsubjpass', 'csubj'}]
            subject_start = min((t.i for t in subjects), default=predicate.i)
            coordinators = [t for t in doc[sent.start:subject_start]
                            if t.dep_ == 'cc' and t.lower_ in {'and', 'but', 'or', 'yet'}
                            and t.i > predicate.head.i]
            if not subjects or not coordinators:
                continue
            if any(t.dep_ in {'advcl', 'relcl', 'ccomp', 'xcomp'} for t in predicate.ancestors):
                continue
            coordinator = min(coordinators, key=lambda t: t.i)
            if coordinator.i <= sent.start or coordinator.i >= min(t.i for t in subjects):
                continue
            if not any(t.is_alpha and not is_filler(t.text) for t in doc[sent.start:coordinator.i]):
                continue
            cuts[coordinator.idx] = 'coordinate_clause_with_subject'
        # ASR fragments before "but" can defeat the full-turn parser. Parse
        # the right clause independently and require a subject + predicate.
        for conjunction in sent:
            if conjunction.lower_ != 'but' or conjunction.i == sent.start:
                continue
            if any(t.dep_ in {'advcl', 'relcl', 'ccomp', 'xcomp'}
                   for t in [conjunction.head, *conjunction.head.ancestors]):
                continue
            right = nlp(text[conjunction.idx + len(conjunction.text):sent.end_char])
            roots = [t for t in right if t.dep_ == 'ROOT' and t.pos_ in {'VERB', 'AUX'}]
            if any(any(c.dep_ in {'nsubj', 'nsubjpass'} for c in root.children) for root in roots):
                cuts[conjunction.idx] = 'contrast_clause_with_subject'
    # A parser sentence edge can fall inside a hyphenated word ("uh-huh").
    # Boundary prediction must preserve the input word stream.
    interiors={i for word in WORD.finditer(text) for i in range(word.start()+1,word.end())}
    positions = sorted(i for i in cuts if i not in interiors)
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
    removed = [w for w in words if is_filler(w)]
    counted = [w for w in words if not is_filler(w)]
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

    Word spelling retained with curly apostrophes normalized; punctuation normalized. Full original text and
    source envelopes remain in the prediction JSON. Do not infer disfluency codes.
    """
    methods=', '.join(sorted({u.get('method',METHOD) for u in utterances}))
    lines = ["@UTF8", "@Begin", "@Languages:\teng",
             "@Participants:\tPAR Target Participant",
             "@ID:\teng|speechpipeline|PAR|||||Participant|||",
             f"@Comment:\tSource: {source}",
             f"@Comment:\tAutomatic unreviewed predictions: {methods}.",
             "@Comment:\tText only; provisional boundaries and word grouping; no MOR review.",
             "@Comment:\tListed fillers removed for word metrics; filler-only units omitted from MLU.",
             "@Comment:\tInternal repetitions unresolved; no retrace codes invented."]
    for utt in utterances:
        text = utt["text"]
        if any(c in text for c in "[]<>@&+*=|\\\x15") or "//" in text:
            raise ValueError(f"Unsupported source annotation in {source}, utterance {utt['utterance_id']}")
        scoring = score_utterance(text)
        if not scoring['include_mlu']:
            continue
        # Match the surface metric policy while leaving source utterances intact.
        words = [word.replace('’', "'") for word in scoring['counted_words']]
        ending = "?" if text.rstrip().endswith("?") else "!" if text.rstrip().endswith("!") else "."
        lines.append("*PAR:\t" + " ".join(words) + " " + ending)
    return "\n".join(lines + ["@End", ""])


# Periods inside these forms are spelling, not annotation delimiters.
ABBREVIATION = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.", re.I)
INITIALISM = re.compile(r"\b(?:[A-Za-z]\.){2,}")


def split_label_cell(value):
    if value is None:
        return []
    text = str(value).strip()
    protected = set()
    for pattern in (ABBREVIATION, INITIALISM):
        for match in pattern.finditer(text):
            protected.update(i for i in range(match.start(), match.end()) if text[i] == '.')
    cuts = [0]
    for match in re.finditer(r"[.!?]+[\"'’”)]*(?=\s|$)|\n+", text):
        # Ellipses mark hesitation/abandonment, not multiple utterances.
        punctuation = match.group().rstrip('"\'’”)')
        if match.start() in protected or punctuation.startswith('..'):
            continue
        cuts.append(match.end())
    cuts.append(len(text))
    return [body for a, b in zip(cuts, cuts[1:]) if (body := text[a:b].strip())]


def flatten_labels(sheet, col=4):
    result = defaultdict(list)
    for row in list(sheet.values)[1:]:
        if len(row) > col and row[2] and row[col]:
            result[str(row[2]).strip()].extend(split_label_cell(row[col]))
    return result


def align(a, b):
    # Global minimum-edit alignment. Exact match, substitution, deletion, insertion tie order.
    if a == b:
        return {i:i for i in range(len(a))},dict(substitutions=0,deletions=0,insertions=0)
    d=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
    d[:,0]=np.arange(len(a)+1)
    d[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i,j]=min(d[i-1,j-1]+(a[i-1]!=b[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    i,j=len(a),len(b)
    matches=[]
    counts=dict(substitutions=0,deletions=0,insertions=0)
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]==b[j-1]: matches.append((i-1,j-1))
            else: counts['substitutions']+=1
            i-=1; j-=1
        elif i and d[i,j]==d[i-1,j]+1:
            counts['deletions']+=1; i-=1
        else:
            counts['insertions']+=1; j-=1
    return dict(reversed(matches)),counts


def plain(text):
    return re.sub(r'\*unintelligible\*', ' xxx ', text, flags=re.I)


def stream(utterances):
    tokens, boundaries = [], set()
    for u in utterances:
        words = [w.casefold().replace('’', "'") for w in WORD.findall(plain(u['text']))]
        if words:
            tokens.extend(words)
            boundaries.add(len(tokens))
    boundaries.discard(len(tokens))
    return tokens, boundaries


def boundary_counts(reference, predicted):
    a, ab = stream(reference)
    b, bb = stream(predicted)
    mapping, edits = align(a, b)
    counts = dict(tp=0, tn=0, fn=0, fp=0)
    for i in range(len(a)-1):
        j = mapping.get(i)
        if j is None or mapping.get(i+1) != j+1 or {a[i], a[i+1]} & {'xxx', 'yyy', 'www'}:
            continue
        truth, pred = i+1 in ab, j+1 in bb
        counts['tp' if truth and pred else 'fn' if truth else 'fp' if pred else 'tn'] += 1
    return dict(**counts, reference_tokens=len(a), pipeline_tokens=len(b),
                reference_possible_gaps=max(0, len(a)-1), pipeline_possible_gaps=max(0, len(b)-1),
                word_edits=sum(edits.values()))


def boundary_scores(c):
    tp, tn, fn, fp = (c[k] for k in ('tp', 'tn', 'fn', 'fp'))
    n = tp+tn+fn+fp
    divide = lambda x,y: x/y if y else None
    po = divide(tp+tn,n)
    pe = divide((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn),n*n)
    return dict(boundary_f1=divide(2*tp,2*tp+fp+fn), precision=divide(tp,tp+fp),
                recall=divide(tp,tp+fn), kappa=divide(po-pe,1-pe) if n else None,
                gap_agreement=po, reference_gap_coverage=divide(n,c['reference_possible_gaps']),
                pipeline_gap_coverage=divide(n,c['pipeline_possible_gaps']),
                word_disagreement=divide(c['word_edits'],c['reference_tokens']))
