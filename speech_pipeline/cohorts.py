"""Preserve participant cohorts and export independently loadable split files.

``eval`` means validation/model selection. Existing test data are historically
exposed: physical separation preserves their role, not pristine test status.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from speech_pipeline import ROOT
from speech_pipeline.pipeline import digest, save

BASE = ROOT/'outputs/validation/benchmark'
BUNDLE = ROOT/'outputs/validation/cohorts_v1'
SPLITS = ('train', 'validation', 'test')
CORPORA = ('ocsc', 'nyu_emerson')
DIRECTORIES = dict(train='train', validation='eval', test='test')


def partition_lookup(groups):
    if set(groups) != set(SPLITS):
        raise ValueError('Require exactly train, validation and test cohorts')
    lookup = {}
    for split in SPLITS:
        for participant in groups[split]:
            if participant in lookup:
                raise ValueError('Duplicate participant or participant leakage between cohorts')
            if participant.split('/')[0] not in CORPORA:
                raise ValueError('Only registered OCSC/NYU participants may enter these cohorts')
            lookup[participant] = split
    return lookup


def validate_records(rows, groups, allowed_splits=SPLITS):
    lookup = partition_lookup(groups)
    seen = set(); owners = {}; source_owners = {}
    for row in rows:
        participant = row['participant']
        if lookup.get(participant) not in allowed_splits:
            raise ValueError('Record belongs to an unknown or forbidden cohort')
        if row['corpus'] != participant.split('/')[0]:
            raise ValueError('Participant/corpus mismatch')
        key = row['corpus'], row['recording']
        if key in seen:
            raise ValueError('Duplicate recording section')
        seen.add(key)
        source = row['corpus'], row['recording'].split('#section')[0]
        if owners.setdefault(source, participant) != participant:
            raise ValueError('One source recording has multiple participant identities')
        sha = row.get('source_sha256')
        if sha and source_owners.setdefault(sha, participant) != participant:
            raise ValueError('Identical source file assigned to multiple participants')
        if row.get('split', lookup[participant]) != lookup[participant]:
            raise ValueError('Record split conflicts with frozen participant assignment')
    return lookup


def validate_training_views(views, train):
    """Every original and crop must come from the supplied training cohort."""
    if views[:len(train)] != train:
        raise ValueError('Training-cache originals differ from frozen training records')
    parents = {(r['corpus'], r['participant'], r['recording']): r for r in train}
    for row in views[len(train):]:
        parent_key = row['corpus'], row['participant'], row['recording'].rsplit('#crop', 1)[0]
        parent = parents.get(parent_key)
        if parent is None or '#crop' not in row['recording']:
            raise ValueError('Augmented record has no parent in the training cohort')
        tokens = row['tokens']; labels = row['labels']; n = len(tokens)
        if not n or len(labels) != n or labels[-1] != -100:
            raise ValueError('Invalid training crop or unmasked final gap')
        if not any(parent['tokens'][i:i+n] == tokens and
                   parent['labels'][i:i+n-1] == labels[:-1]
                   for i in range(len(parent['tokens'])-n+1)):
            raise ValueError('Training crop words/labels differ from their training parent')


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(destination=BUNDLE):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve the frozen cohort bundle; verify it instead')
    plan = json.loads((BASE/'metric_training.json').read_text())
    groups = plan['partitions']; lookup = partition_lookup(groups)
    # Verify that all historical rounds retain the same participant assignment.
    plans = ['talkbank_split.json', 'improvement_protocol.json', 'reassessment.json', 'metric_training.json']
    evidence = {}
    for name in plans:
        path = BASE/name; document = json.loads(path.read_text())
        previous = document.get('partitions', document.get('participants'))
        if {s: set(ps) for s, ps in previous.items()} != {s: set(ps) for s, ps in groups.items()}:
            raise ValueError('Historical cohort assignments disagree')
        evidence[str(path.relative_to(ROOT))] = digest(path)
    rows = []
    for name, field, allowed in [
        ('talkbank_spoken_development.jsonl', 'development_data_sha256', ('train', 'validation')),
        ('talkbank_spoken_test.jsonl', 'test_data_sha256', ('test',))]:
        path = BASE/name
        if digest(path) != plan[field]:
            raise ValueError('Prepared dataset changed')
        selected = _read_jsonl(path); validate_records(selected, groups, allowed)
        rows.extend(selected); evidence[str(path.relative_to(ROOT))] = digest(path)
    validate_records(rows, groups)
    sources = {}
    texts = defaultdict(list)
    for row in rows:
        source = ROOT/'dataset/public'/row['corpus']/row['recording'].split('#section')[0]
        if source not in sources:
            actual = digest(source)
            if actual != row['source_sha256']:
                raise ValueError('Original source changed')
            sources[source] = actual
        elif sources[source] != row['source_sha256']:
            raise ValueError('Inconsistent source hashes')
        fingerprint = hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest()
        texts[fingerprint].append(dict(corpus=row['corpus'], participant=row['participant'],
            recording=row['recording'], split=lookup[row['participant']], words=len(row['tokens'])))
    duplicate_texts = [v for v in texts.values() if len({r['split'] for r in v}) > 1]
    if duplicate_texts:
        raise ValueError('Exact section text crosses cohorts; review before exporting')
    # These retained caches were actually used by the two training rounds.
    cache_audit = []
    for path, originals in [
        (Path('/private/tmp/speech_metric_research/spoken_augmented_training.jsonl'),
         [r for r in rows if lookup[r['participant']] == 'train']),
        (Path('/private/tmp/speech_reassessment/augmented_training.jsonl'),
         [r for r in _read_jsonl(BASE/'talkbank_training.jsonl') if lookup[r['participant']] == 'train'])]:
        if path.exists():
            views = _read_jsonl(path); validate_training_views(views, originals)
            cache_audit.append(dict(path=str(path), sha256=digest(path), records=len(views),
                allowed_split='train', all_parents_and_labels_verified=True))
        else:
            cache_audit.append(dict(path=str(path), status='not_available_for_audit'))
    selection = json.loads((ROOT/'outputs/results/transcript_validation/chat_calibration_validation.json').read_text())
    if any(lookup.get(r['participant']) != 'validation' for r in selection['records']):
        raise ValueError('Non-validation record in calibration selection')
    parity = json.loads((ROOT/'outputs/validation/transcript_validation/calibration_contexts.json').read_text())
    if any(lookup.get(r['participant']) != 'validation' or r['split'] != 'validation' for r in parity):
        raise ValueError('Non-validation context in pre-selection calibration cache')
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        assignments=groups, eval_means='validation: model/workflow/calibration selection only',
        policy=dict(train='Gradient updates and fitted preprocessing; training-derived augmentation only',
            validation='Model/epoch/workflow/margin selection; no gradient updates',
            test='Frozen evaluation only; no tuning, augmentation, fitting or reassignment',
            test_exposure='Historically exposed descriptive test, not a pristine independent test. Do not reshuffle to erase exposure.',
            lucid='Separate historically exposed lexical benchmark; absent from these ML cohorts',
            original_ten_participants='Excluded from this cohort bundle; historical evaluation only'),
        source_evidence=evidence, files={}, record_order=[[r['corpus'], r['recording']] for r in rows],
        audit=dict(participant_overlap=0, recording_overlap=0, source_file_overlap=0,
            exact_section_text_overlap=0, records=len(rows), source_files=len(sources),
            training_caches=cache_audit, calibration_selection_validation_only=True,
            calibration_preselection_contexts_validation_only=True),
        exclusions=json.loads((ROOT/'outputs/validation/transcript_validation/unscored_source_reasons.json').read_text()))
    # All checks finish before writing; raw datasets and prior outputs are untouched.
    destination.mkdir(parents=True)
    for split in SPLITS:
        manifest['files'][split] = {}
        for corpus in CORPORA:
            selected = [r for r in rows if r['corpus'] == corpus and lookup[r['participant']] == split]
            relative = Path(DIRECTORIES[split])/f'{corpus}.jsonl'; path = destination/relative
            path.parent.mkdir(exist_ok=True)
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
            manifest['files'][split][corpus] = dict(path=str(relative), sha256=digest(path),
                records=len(selected), participants=len({r['participant'] for r in selected}),
                assigned_participants=sum(p.startswith(corpus+'/') for p in groups[split]))
    save(destination/'manifest.json', manifest)
    (destination/'manifest.sha256').write_text(digest(destination/'manifest.json')+'\n')
    (destination/'README.md').write_text(
        '# Frozen participant cohorts\n\n'
        'train/: training only. eval/: validation/model selection only. test/: frozen evaluation only.\n'
        'Each directory has separate OCSC and NYU files. All sections and crops from a child retain the same role.\n'
        'LUCID is a separate lexical benchmark and is not a segmentation-training cohort.\n\n'
        'The original assignments are preserved. Two assigned NYU training children have no eligible sections; '
        'two OCSC files have no selected discourse tasks. Exclusions remain in manifest.json.\n\n'
        'The existing test cohort is historically exposed. These exports do not make it a fresh independent test. '
        'Do not reshuffle, fit on test, or use test errors to select changes.\n\n'
        'Load one role with speech_pipeline.cohorts.load_split("train"), load_split("validation"), '
        'or load_split("test"). Development code must explicitly request only train/validation. '
        'Each load checks the requested files and participant assignments without opening other split files.\n'
        'Verify the complete bundle with .venv/bin/python -m speech_pipeline.cohorts verify.\n')
    verify(destination)
    return manifest


def read_manifest(bundle=BUNDLE):
    bundle = Path(bundle)
    if digest(bundle/'manifest.json') != (bundle/'manifest.sha256').read_text().strip():
        raise ValueError('Frozen cohort manifest changed')
    return json.loads((bundle/'manifest.json').read_text())


def load_split(split, bundle=BUNDLE, expected_assignments=None):
    if split == 'eval':split = 'validation'
    if split not in SPLITS:raise ValueError('Request one explicit cohort: train, validation/eval or test')
    bundle = Path(bundle); manifest = read_manifest(bundle)
    if expected_assignments is not None and manifest['assignments'] != expected_assignments:
        raise ValueError('Cohort bundle differs from registered participant assignments')
    rows = []
    for corpus, info in manifest['files'][split].items():
        path = bundle/info['path']
        if not path.resolve().is_relative_to((bundle/DIRECTORIES[split]).resolve()):
            raise ValueError('Cohort file is outside its assigned split directory')
        if digest(path) != info['sha256']:raise ValueError('Frozen cohort file changed')
        selected = _read_jsonl(path)
        if any(r['corpus'] != corpus for r in selected):raise ValueError('Wrong corpus in cohort file')
        validate_records(selected, manifest['assignments'], (split,))
        if len(selected) != info['records']:raise ValueError('Cohort count changed')
        rows.extend(selected)
    validate_records(rows, manifest['assignments'], (split,))
    order = {tuple(key): i for i, key in enumerate(manifest['record_order'])}
    if any((r['corpus'], r['recording']) not in order for r in rows):
        raise ValueError('Unknown section in frozen cohort')
    return sorted(rows, key=lambda r: order[r['corpus'], r['recording']])


def verify(bundle=BUNDLE):
    bundle = Path(bundle); manifest = read_manifest(bundle)
    rows = [r for split in SPLITS for r in load_split(split, bundle)]
    validate_records(rows, manifest['assignments'])
    if len(rows) != manifest['audit']['records']:raise ValueError('Incomplete cohort coverage')
    return dict(records=len(rows), participants=len({r['participant'] for r in rows}),
        participant_overlap=0, assignments_preserved=True, file_hashes_verified=True,
        test_status=manifest['policy']['test_exposure'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'verify'])
    args = parser.parse_args()
    result = build() if args.stage == 'build' else verify()
    print(json.dumps(result.get('audit', result), indent=2))


if __name__ == '__main__':main()
