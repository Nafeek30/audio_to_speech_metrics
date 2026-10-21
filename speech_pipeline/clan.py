"""Apply the current utterance splitter to corrected transcripts for a CLAN run.

Run: .venv/bin/python -m speech_pipeline.clan
This separate export does not change the saved publication comparisons.
"""
import argparse
from bisect import bisect_left
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil

import spacy
import speech_pipeline.utterances as auto
from speech_pipeline.utterances import align
from speech_pipeline.pipeline import get_corrected_filename, SOURCE_FILES

from speech_pipeline import ROOT
DEFAULT_OUT = ROOT / 'outputs/validation/clan/same_boundaries'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def corrected_turns(rows, speaker, corrected_text):
    """Map source speaker-turn breaks, never source utterance predictions.

    The corrected files contain target speech without interlocutor tiers.
    Exact word anchors in a minimum-edit alignment locate the source turns.
    Between nonadjacent anchors, interpolate the turn break and flag it for
    review. Corrected line breaks are whitespace, not mandatory utterances.
    """
    text = re.sub(r'\s+', ' ', corrected_text).strip()
    turns = list(auto.target_turns(rows, speaker))
    raw_words, raw_breaks = [], [0]
    for turn in turns:
        words = auto.WORD.findall(' '.join(row['text'] for row in turn))
        if not words:
            raise ValueError('Source turn has no words; review speaker-turn mapping')
        raw_words.extend(words)
        raw_breaks.append(len(raw_words))
    words = list(auto.WORD.finditer(text))
    normalize = lambda w: w.casefold().replace('’', "'")
    anchors, _ = align([normalize(w) for w in raw_words],
                       [normalize(w.group()) for w in words])
    if not anchors:
        raise ValueError('No exact word anchors for corrected speaker turns')
    indices = list(anchors)
    cuts, audit = [0], []
    for gap in raw_breaks[1:-1]:
        at = bisect_left(indices, gap)
        li = indices[at - 1] if at else -1
        ri = indices[at] if at < len(indices) else len(raw_words)
        lj = anchors[li] if li >= 0 else -1
        rj = anchors[ri] if ri < len(raw_words) else len(words)
        left, right = li + 1, ri
        mapped = rj if left == right else round(lj + 1 +
                    (gap - left) * (rj - lj - 1) / (right - left))
        mapped = max(cuts[-1], min(len(words), mapped))
        cuts.append(mapped)
        audit.append(dict(source_word_gap=gap, corrected_word_gap=mapped,
                          left_source_anchor=li, right_source_anchor=ri,
                          left_corrected_anchor=lj, right_corrected_anchor=rj,
                          review_required=not (li == gap - 1 and ri == gap and rj == lj + 1)))
    cuts.append(len(words))
    predicted_turns = []
    for i, (first, last) in enumerate(zip(cuts, cuts[1:])):
        if first == last:
            continue  # A source turn deleted in the corrected text has no text to score.
        begin = words[first].start() if first else 0
        end = words[last].start() if last < len(words) else len(text)
        predicted_turns.append(dict(turn_id=i + 1, text=text[begin:end].strip(),
                                    corrected_char_start=begin,
                                    source_rows=[r['row'] for r in turns[i]],
                                    turn_mapping_review_required=(i > 0 and audit[i-1]['review_required']) or
                                        (i < len(audit) and audit[i]['review_required'])))
    assert ''.join(''.join(t['text'].split()) for t in predicted_turns) == ''.join(text.split())
    return text, predicted_turns, audit, len(anchors) / len(words)


def predict_corrected(rows, speaker, corrected_text, nlp):
    """Use the same corrected segmentation for exports and pipeline scoring."""
    text, turns, audit, coverage = corrected_turns(rows, speaker, corrected_text)
    utterances = []
    for turn in turns:
        for start, end, body, reason in auto.split_turn(turn['text'], nlp):
            utterances.append(dict(utterance_id=len(utterances) + 1, text=body,
                turn_id=turn['turn_id'], source_rows=turn['source_rows'],
                corrected_char_start=turn['corrected_char_start'] + start,
                corrected_char_end=turn['corrected_char_start'] + end,
                boundary_reason=reason,
                turn_mapping_review_required=turn['turn_mapping_review_required'],
                scoring=auto.score_utterance(body)))
    assert ''.join(''.join(u['text'].split()) for u in utterances) == ''.join(text.split())
    return text, utterances, audit, coverage


def build(out):
    nlp = spacy.load('en_core_web_sm')
    prior = ROOT / 'outputs/validation/method_comparison/analysis/uncorrected'
    report = json.loads((prior / 'comparison.json').read_text())
    if report['method'] != auto.METHOD:
        raise ValueError('Saved uncorrected results do not use the current utterance method')
    for folder in ['corrected_auto', 'uncorrected_auto', 'utterances']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest, full = [], []
    for rec in report['recordings']:
        pid, name, speaker = (rec[k] for k in ['patient_id', 'source_file', 'target_speaker'])
        raw = ROOT / 'dataset/uncorrected' / pid / name
        corrected = ROOT / 'dataset/corrected' / pid / get_corrected_filename(pid, name)
        if digest(raw) != rec['raw_sha256']:
            raise ValueError(f'Uncorrected source changed: {raw}')
        rows = list(auto.read_rows(raw))
        saved_predictions = json.loads((prior / rec['utterance_file']).read_text())['utterances']
        if auto.predict(rows, speaker, nlp) != saved_predictions:
            raise ValueError(f'Predictions changed since the saved uncorrected run: {raw}')
        text, utterances, turn_audit, coverage = predict_corrected(rows, speaker,
                                                               corrected.read_text(encoding='utf-8'), nlp)
        filename = f'{pid}_{Path(name).stem}.cha'
        content = auto.chat_text(utterances, f'{pid}/{corrected.name}; speaker {speaker}')
        content = content.replace('@Comment:\tAutomatic unreviewed predictions:',
                                  '@Comment:\tCorrected text; automatic utterance predictions:')
        content = content.replace('@Comment:\tText only;',
                                  '@Comment:\tSource speaker turns mapped by word alignment. Text only;')
        content = content.replace('’', "'")
        (out / 'corrected_auto' / filename).write_text(content, encoding='utf-8')
        metrics = auto.word_mlu(utterances)
        assert sum(s.startswith('*PAR:') for s in content.splitlines()) == metrics['mlu_nutterances']
        old = ROOT / 'outputs/validation/clan/same_boundaries/uncorrected_auto' / filename
        saved = prior / rec['chat_file']
        assert [s.replace('’', "'") for s in saved.read_text().splitlines() if s.startswith('*PAR:')] == [
            s for s in old.read_text().splitlines() if s.startswith('*PAR:')]
        shutil.copy2(old, out / 'uncorrected_auto' / filename)
        jsonname = Path(filename).with_suffix('.json').name
        item = dict(patient_id=pid,uncorrected_source=name,corrected_source=corrected.name,
                    target_speaker=speaker,chat_filename=filename,method=auto.METHOD,
                    raw_sha256=digest(raw),corrected_sha256=digest(corrected),
                    normalized_corrected_text=text,utterances=utterances,turn_alignment=turn_audit,
                    corrected_exact_word_anchor_fraction=coverage,corrected_metrics=metrics,
                    uncorrected_metrics=rec['mlu'])
        (out / 'utterances' / jsonname).write_text(json.dumps(item, indent=2, ensure_ascii=False)+'\n')
        full.append(item)
        manifest.append(dict(patient_id=pid,chat_filename=filename,uncorrected_source=name,
            corrected_source=corrected.name,target_speaker=speaker,
            corrected_total_utterances=metrics['total_utterances'],
            corrected_mlu_utterances=metrics['mlu_nutterances'],corrected_mlu_words=metrics['mlu_nwords'],
            corrected_mlu_words_mean=metrics['mlu_words'],corrected_mlu_words_sd=metrics['mlu_words_sd'],
            uncorrected_total_utterances=rec['mlu']['total_utterances'],
            uncorrected_mlu_utterances=rec['mlu']['mlu_nutterances'],uncorrected_mlu_words=rec['mlu']['mlu_nwords'],
            uncorrected_mlu_words_mean=rec['mlu']['mlu_words'],uncorrected_mlu_words_sd=rec['mlu']['mlu_words_sd'],
            corrected_exact_word_anchor_fraction=coverage,
            uncertain_turn_breaks=sum(a['review_required'] for a in turn_audit),
            corrected_chat_sha256=digest(out/'corrected_auto'/filename),
            uncorrected_chat_sha256=digest(out/'uncorrected_auto'/filename),
            utterance_json=f'utterances/{jsonname}'))
        print(f'{pid}/{corrected.name}: {len(utterances)} utterances; {metrics["mlu_nutterances"]} MLU-eligible',flush=True)
    write_csv(out/'manifest.csv',manifest)
    (out/'provenance.json').write_text(json.dumps(dict(method=auto.METHOD,
        code_sha256={p:digest(ROOT/p) for p in SOURCE_FILES},
        versions={p:version(p) for p in ['spacy','en-core-web-sm']},
        uncorrected_report_sha256=digest(prior/'comparison.json'),
        corrected_turn_policy='Original target speaker turns mapped by minimum-edit exact-word anchors; interpolate uncertain breaks; ignore corrected line breaks as boundaries',
        corrected_boundary_policy='speech_pipeline.utterances.split_turn applied within mapped source turns',
        publication_results_changed=False),indent=2)+'\n')
    return full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.output_dir)
