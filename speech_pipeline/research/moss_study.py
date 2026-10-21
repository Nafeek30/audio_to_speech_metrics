"""Register audio/CHAT supervision against the existing immutable child cohorts.

This preparation command never trains or scores a model. Reference-only JSONL
files deliberately are not directly consumable by a model trainer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from speech_pipeline import ROOT
from speech_pipeline.audio_download import now, prepare_plan, save, sha_file
from speech_pipeline.cohorts import BUNDLE, DIRECTORIES, SPLITS, load_split, read_manifest, verify
from speech_pipeline.corpora import read_chat


def check_folds(folds, pool):
    """Require a complete, disjoint five-fold partition of original train."""
    pool = set(pool)
    heldout = Counter()
    if len(folds) != 5:
        raise ValueError('Require the five existing folds')
    if {fold['fold'] for fold in folds} != {1, 2, 3, 4, 5}:
        raise ValueError('Require unique original fold numbers 1 through 5')
    for fold in folds:
        parts = []
        for role in ('fit', 'inner', 'outer'):
            values = fold[role]
            if len(values) != len(set(values)) or not values:
                raise ValueError('Empty or repeated CV participant')
            parts.append(set(values))
        fit, inner, outer = parts
        if fit & inner or fit & outer or inner & outer:
            raise ValueError('CV participant leakage')
        if fit | inner | outer != pool:
            raise ValueError('CV must use original training participants only')
        heldout.update(outer)
    if set(heldout) != pool or any(n != 1 for n in heldout.values()):
        raise ValueError('Every training participant must be held out once')


def label_flags(row):
    flags = []
    times = row['time_marks']
    if (not times or len(times) != 2 or
            any(not isinstance(t, (int, float)) for t in times) or
            not 0 <= times[0] < times[1]):
        flags.append('missing_or_invalid_timing')
    if re.search(r'\b(?:xxx|yyy|www)\b', row['text'], re.I):
        flags.append('unintelligible_or_untranscribed_speech')
    if re.search(r'\b\w*(?:Childname|Mothername|Investigatorname)\w*\b', row['text'], re.I):
        flags.append('anonymized_spoken_name')
    if not row['text']:
        flags.append('nonlexical_main_tier')
    return flags


def section_inventory(record, split, rows, source):
    sections = []
    for row in rows:
        if not sections or sections[-1][0]['gem'] != row['gem']:
            sections.append([])
        sections[-1].append(row)
    index = int(record['recording'].rsplit('#section', 1)[1]) - 1
    selected = sections[index]
    if selected[0]['gem'] != record['gem']:
        raise ValueError('Frozen section label no longer matches source CHAT')
    labels = [dict(source_row=r['row'], chat_speaker=r['speaker'],
        spoken_words=r['text'], audible=r['audible'], raw_chat=r['raw_chat'],
        time_marks_ms=r['time_marks'], flags=label_flags(r)) for r in selected]
    counts = Counter(flag for row in labels for flag in row['flags'])
    return dict(corpus=record['corpus'], participant=record['participant'],
        cohort=split, recording=record['recording'], gem=record['gem'],
        source=str(source), source_sha256=record['source_sha256'],
        purpose='Reference labels only; never an inference prompt or a model input',
        labels=labels, quality_flag_counts=dict(counts),
        timing_provenance='Source CHAT alignment; automatic aligner comments retained; not assumed manually verified')


def register(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve an existing registration; select a new destination')
    verification = verify()
    manifest = read_manifest()
    cv_path = ROOT/'outputs/validation/participant_cv_v1/protocol.json'
    cv = json.loads(cv_path.read_text())
    cohort_sha = sha_file(BUNDLE/'manifest.json')
    if cv['original_cohort_manifest_sha256'] != cohort_sha:
        raise ValueError('Original CV belongs to a different cohort manifest')
    records = {split: load_split(split) for split in SPLITS}
    pool = {r['participant'] for r in records['train']}
    check_folds(cv['folds'], pool)
    plan = prepare_plan()
    jobs = [j for j in plan['jobs'] if j['source'] in ('ocsc', 'nyu_emerson')]
    job_lookup = {(j['source'], j['transcript']): j for j in jobs}
    parsed = {}
    headers = {}
    inventories = {}
    for split in SPLITS:
        inventories[split] = []
        for record in records[split]:
            relative = record['recording'].split('#section')[0]
            source = ROOT/'dataset/public'/record['corpus']/relative
            job = job_lookup[record['corpus'], relative]
            if (job['participant'], job['cohort']) != (record['participant'], split):
                raise ValueError('Audio and reference cohorts disagree')
            if source not in parsed:
                if sha_file(source) != record['source_sha256']:
                    raise ValueError('Original CHAT changed')
                parsed[source] = read_chat(source)
                headers[str(source)] = [s for s in source.read_text(encoding='utf-8-sig').splitlines()
                    if s.startswith('@Comment:') and re.search(r'align|morph|batchalign', s, re.I)]
            inventories[split].append(section_inventory(record, split, parsed[source], source))
    # Complete all validation before publishing any registration files.
    destination.mkdir(parents=True)
    files = {}
    for split, rows in inventories.items():
        folder = destination/DIRECTORIES[split]
        folder.mkdir()
        for corpus in ('ocsc', 'nyu_emerson'):
            path = folder/f'{corpus}.references.jsonl'
            selected = [r for r in rows if r['corpus'] == corpus]
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
            files[str(path.relative_to(destination))] = dict(sha256=sha_file(path), sections=len(selected),
                participants=len({r['participant'] for r in selected}),
                quality_flag_counts=dict(sum((Counter(r['quality_flag_counts']) for r in selected), Counter())))
    save(destination/'audio_jobs.json', dict(jobs=jobs, frozen_cohort_manifest_sha256=cohort_sha))
    save(destination/'source_processing_comments.json', headers)
    protocol = dict(schema_version=1, registered_utc=now(),
        authorization='User requested matching audio acquisition and a NEW MOSS audio/CHAT fine-tuning study on 2026-10-20.',
        frozen_cohort_manifest_sha256=cohort_sha, verification=verification,
        files=files, recordings=len(jobs), sections=sum(map(len, inventories.values())),
        original_cv_protocol=str(cv_path), original_cv_protocol_sha256=sha_file(cv_path), folds=cv['folds'],
        split_policy=manifest['policy'],
        input_policy='Audio and fixed transcription instruction only. CHAT text, roles, timestamps and task labels are reference targets, not inference features.',
        supervision_policy='Retain every source tier and raw CHAT file. Train on all audible speakers with anonymous IDs assigned by first occurrence within each audio window. Never use CHI as a privileged predicted speaker ID.',
        quality_policy='Audit unresolved timing, anonymization, nonlexical events, unintelligibility, continuation and overlap before exporting training windows. Never fabricate speech, words or time labels. Retain full eval/test coverage and explicitly report unavailable scores.',
        selection_policy='Five original-training folds, fresh pinned base initialization each fit. Inner participants select epoch; paired outer predictions compare preregistered recipes. Original validation selects final epoch from original-train fit. Freeze predictions/recipe before one descriptive test evaluation. No test-driven changes.',
        objective='Primary: corpus-balanced participant-mean speaker-permutation WER. Also pooled WER, cpWER, child WER with explicitly disclosed role assignment; native CHAT/CLAN utterance count, MLUw and length SD. Report each corpus and coverage; do not combine per-metric minima across models.',
        test_caveat=manifest['policy']['test_exposure'],
        previous_pilot='OCSC 4002 is an existing exposed test feasibility pilot; excluded from gradient updates and all tuning.',
        old_experiments='Original transcript model, weights and cv_hyperparameter_v4/user_stop.json remain preserved. This registration does not resume them.',
        official_finetuning_guide='https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md',
        state='registered_reference_inventory_only; audio acquisition, audited window preparation and training pending')
    save(destination/'protocol.json', protocol)
    (destination/'protocol.sha256').write_text(sha_file(destination/'protocol.json')+'\n')
    return {k: protocol[k] for k in ('recordings', 'sections', 'verification', 'state')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'outputs/validation/moss_audio_v1/registration')
    args = parser.parse_args()
    print(json.dumps(register(args.destination), indent=2))


if __name__ == '__main__':
    main()
