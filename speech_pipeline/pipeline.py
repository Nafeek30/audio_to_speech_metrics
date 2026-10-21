"""Generate reproducible predicted utterances, TAALED comparison and CLAN inputs.

Run from the project root: .venv/bin/python -m speech_pipeline
After running CLAN on the generated files, use --clan-results DIRECTORY.
Old CLAN results cannot match the content-hashed filenames for this run.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import spacy
import speech_pipeline.utterances as auto
import speech_pipeline.metrics as custom
import speech_pipeline.taaled as official
from speech_pipeline.export import export_all

from speech_pipeline import ROOT
OUT = ROOT / "outputs/results/pipeline"
SOURCE_FILES = tuple(f"speech_pipeline/{name}.py" for name in
                     ("utterances", "pipeline", "metrics", "taaled", "clan", "export"))
METRICS = ["basic_ntokens", "basic_ntypes", "basic_ncontent_tokens", "basic_ncontent_types",
    "basic_nfunction_tokens", "basic_nfunction_types", "lexical_density_tokens",
    "lexical_density_types", "simple_ttr_aw", "root_ttr_aw", "log_ttr_aw",
    "maas_ttr_aw", "mattr50_aw", "msttr50_aw", "hdd42_aw", "mtld_original_aw",
    "mtld_mean_factor_length_aw"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def difference(a, b):
    return {"custom": a, "reference": b,
            "difference": a - b if a is not None and b is not None else None,
            "absolute_error": abs(a - b) if a is not None and b is not None else None,
            "absolute_percent_error": 100 * abs(a - b) / abs(b)
                if a is not None and b is not None and b != 0 else None}


def lexical(utterances, nlp, temp):
    text = "\n".join(clean for u in utterances if (clean := auto.analysis_text(u['text'])))
    tokens, tagged, metadata = custom.preprocess_text(text, nlp)
    mine = custom.calculate_speech_metrics(text, tokens, tagged, metadata)
    temp.write_text(text)
    theirs = official.run_taaled(temp)
    return {k: difference(mine[k], theirs[k]) for k in METRICS}


def clan_reference(path, stem, expected):
    text = path.read_text()
    if text.splitlines()[0].strip() != "mlu +t*PAR -t%mor +f":
        raise ValueError(f"Unexpected CLAN command: {path}")
    if f"{stem}.cha>" not in text:
        raise ValueError(f"CLAN source mismatch: {path}")
    counts = re.search(r"utterances = (\d+), words = (\d+)", text)
    mean = re.search(r"Ratio of words over utterances = ([\d.]+)", text)
    sd = re.search(r"Standard deviation = ([\d.]+)", text)
    if not all((counts, mean, sd)):
        raise ValueError(f"Incomplete CLAN result: {path}")
    u, w = map(int, counts.groups())
    if not u or abs(w / u - float(mean[1])) > .000501:
        raise ValueError(f"Inconsistent CLAN result: {path}")
    ref = dict(zip(("mlu_nutterances", "mlu_nwords", "mlu_words", "mlu_words_sd"),
                   (u, w, float(mean[1]), float(sd[1]))))
    return {"status": "software_reference_imported", "output_sha256": digest(path),
            "metrics": {k: difference(expected[k], v) for k, v in ref.items()}}


def get_corrected_filename(patient, uncorrected_filename):
    corr_dir = ROOT / 'dataset/corrected' / patient
    corr_files = sorted(corr_dir.glob('*.txt'))
    u_stem = Path(uncorrected_filename).stem
    for cf in corr_files:
        parts = u_stem.split('_transcription')[0]
        if parts in cf.name:
            return cf.name
    uncorr_dir = ROOT / 'dataset/uncorrected' / patient
    u_files = sorted(uncorr_dir.glob('*.xlsx'))
    idx = [f.name for f in u_files].index(uncorrected_filename)
    return corr_files[idx].name


def resolve_segmenter(setting,nlp,model_directory=None,expected_weights_sha256=None):
    if setting=='rules':return None
    folder=ROOT/'outputs/results/benchmark/metric_segmenter'
    if model_directory is not None:
        folder=(ROOT/model_directory).resolve()
        if not folder.is_relative_to(ROOT.resolve()):raise ValueError('Saved transcript model must be inside the project')
        if (folder/'ensemble.json').exists():
            if expected_weights_sha256 is None or digest(folder/'ensemble.json')!=expected_weights_sha256:
                raise ValueError('Saved transcript workflow/ensemble manifest mismatch')
            from speech_pipeline.ensemble import EnsembleSegmenter
            return EnsembleSegmenter(folder,nlp)
        if not (folder/'segmenter.json').exists():raise FileNotFoundError('The saved transcript model is unavailable')
        if expected_weights_sha256 is None or digest(folder/'model.safetensors')!=expected_weights_sha256:
            raise ValueError('Saved transcript workflow/model weight mismatch')
    else:
        if not (folder/'segmenter.json').exists():folder=ROOT/'outputs/results/benchmark/context_segmenter'
        if not (folder/'segmenter.json').exists():folder=ROOT/'outputs/results/benchmark/segmenter'
    if setting=='auto' and not (folder/'segmenter.json').exists():return None
    if not (folder/'segmenter.json').exists():
        raise FileNotFoundError('The selected trained segmenter is unavailable; use --segmenter rules or finish the research training workflow')
    try:
        from speech_pipeline.segmentation import Segmenter
    except ImportError as e:
        raise RuntimeError('Install segmenter inference dependencies: uv sync --locked --extra segmenter') from e
    return Segmenter(folder,nlp)


def segmenter_evidence(segmenter):
    if segmenter is None:return dict(name=auto.METHOD,kind='rules')
    folder=segmenter.folder
    if (folder/'ensemble.json').exists():
        return dict(name='gap_probability_ensemble',kind='pretrained_ensemble',device=str(segmenter.device),
            selected=segmenter.metadata['name'],decoding=segmenter.metadata['decoding'],
            artifacts={'ensemble.json':digest(folder/'ensemble.json')},members=segmenter.metadata['members'],
            training_input_sha256=segmenter.metadata['input_sha256'])
    return dict(name='pretrained_joint_gap_segmentation',kind='pretrained',device=str(segmenter.device),
        selected=segmenter.metadata['name'],threshold=segmenter.metadata['threshold'],
        decoding=segmenter.metadata.get('decoding','legacy_validation_cutoff'),
        artifacts={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()},
        training_input_sha256=segmenter.metadata['input_sha256'])


def clan_software_check(utterances,temp):
    """Score the identical exported, filler-excluded input with official CLAN."""
    from speech_pipeline.corpora import run_clan
    chat=temp/'metric_check.cha';chat.write_text(auto.chat_text(utterances,'pipeline metric verification'))
    profile=custom.clinical_from_predictions(utterances)['values']
    if profile['TNW']==0:return dict(status='no_eligible_utterances_for_CLAN',metrics={},requested_metrics={})
    mlu=run_clan('mlu',chat)
    freq=run_clan('freq',chat,window=50 if profile['TNW']>=50 else None)
    values={k:freq[k] for k in ('TNW','NDW','TTR','MATTR')};values['MLUw']=mlu['mlu']
    expected=auto.word_mlu(utterances)
    return dict(status='software_reference_computed',
        metrics={k:difference(expected[k],v) for k,v in
                 [('mlu_nutterances',mlu['utterances']),('mlu_nwords',mlu['total']),('mlu_words',mlu['mlu']),('mlu_words_sd',mlu['sd'])]},
        requested_metrics={k:difference(profile[k],v) for k,v in values.items()},
        official_outputs=dict(mlu=mlu,freq=freq),
        scope='Identical normalized predicted text; calculator check, not a gold-boundary validation')


def run_paragraph(path,nlp=None,output_root=None,metric_profile='clinical',segmenter='auto',verify_clan=False):
    path=Path(path);text=path.read_text(encoding='utf-8-sig')
    nlp=nlp if nlp is not None else spacy.load('en_core_web_sm')
    trained=resolve_segmenter(segmenter,nlp);evidence=segmenter_evidence(trained)
    utterances=trained.segment(text) if trained else auto.predict([dict(row=2,speaker='PAR',text=text,start='',end='')],'PAR',nlp)
    if not utterances:raise ValueError('No speech words in the paragraph')
    out=Path(output_root or OUT)/'paragraph'
    with TemporaryDirectory(prefix='paragraph_metrics_') as temp:
        record=dict(patient_id='paragraph',source_file=path.name,target_speaker='PAR',raw_sha256=digest(path),
            source_rows=len(text.splitlines()),prediction_file='segmented_utterances/paragraph.json',
            utterance_file='segmented_utterances/paragraph.json',mlu=auto.word_mlu(utterances),
            lexical=lexical(utterances,nlp,Path(temp)/'lexical.txt'),
            clan=clan_software_check(utterances,Path(temp)) if verify_clan else dict(status='rerun_CLAN_required'))
    body=auto.chat_text(utterances,path.name);sha=hashlib.sha256(body.encode()).hexdigest()
    record.update(chat_file=f'clan_software_input_files/paragraph_auto_{sha[:12]}.cha',chat_sha256=sha)
    if metric_profile=='clinical':record['requested_metrics']=custom.clinical_from_predictions(utterances)
    patient={k:record[k] for k in ('patient_id','mlu','lexical','clan')}
    if metric_profile=='clinical':patient['requested_metrics']=record['requested_metrics']
    policy=dict(auto.POLICY)
    if trained:policy.update(segmentation='Pretrained left/right word-gap model with rule features',
        automatic_boundary_rule='Paragraph-derived rules are joint model features; source row boundaries are not utterance cuts')
    report=dict(run_utc=datetime.now(timezone.utc).isoformat(),dataset_type='paragraph',metric_profile=metric_profile,
        method=evidence['name'],segmenter=evidence,scope='Text-only paragraph segmentation and downstream metrics; no audio or reviewed clinical annotations',
        scoring_policy=policy,recordings=[record],patients=[patient])
    save(out/record['prediction_file'],dict(method=evidence['name'],source_file=str(path.resolve()),raw_sha256=digest(path),utterances=utterances))
    chat=out/record['chat_file'];chat.parent.mkdir(parents=True,exist_ok=True);chat.write_text(body)
    save(out/'comparison.json',report);export_all(report,out)
    print(f'Paragraph: {len(utterances)} utterances; results {out}',flush=True)
    return report


def run_pipeline(dataset_type='uncorrected', nlp=None, output_root=None, corrected_boundary_mode='automatic', metric_profile='standard',segmenter='auto',verify_clan=False,boundary_context='speaker-turns'):
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    mapping = official.load_target_speakers()
    out_dir = Path(output_root or OUT) / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    is_corr = (dataset_type == 'corrected')
    source_lines = is_corr and corrected_boundary_mode == 'source-lines'
    if boundary_context not in {'speaker-turns','paragraph'}:raise ValueError('Unknown boundary context')
    if source_lines and boundary_context=='paragraph':raise ValueError('Source-line references and paragraph prediction are different modes')
    # The pretrained model was selected for continuous paragraphs. Its transfer
    # to diarized workbooks underperformed the established turn-aware rules.
    # Keep that experimental mode explicit instead of changing this default.
    setting=('improved' if boundary_context=='paragraph' else 'rules') if segmenter=='auto' else segmenter
    trained=None if source_lines else resolve_segmenter(setting,nlp)
    if boundary_context=='paragraph' and trained is None:raise ValueError('Paragraph workbook mode requires a trained model; use the plain --paragraph command for the legacy rules baseline')
    evidence=segmenter_evidence(trained)
    method_name = "corrected_transcript_source_lines" if source_lines else evidence['name']
    scope_desc = ("Corrected transcripts with each nonempty source line retained as an utterance"
                  if source_lines else
                  "Corrected transcripts with automatic clauses; source turns mapped by word alignment"
                  if is_corr else
                  "Uncorrected ASR transcripts with automatic utterance segmentation")
    policy=dict(auto.POLICY)
    if trained:
        scope_desc=('Corrected' if is_corr else 'Uncorrected ASR')+' transcripts; pretrained segmentation within observed/mapped speaker turns'
        if boundary_context=='paragraph':scope_desc=('Corrected' if is_corr else 'Uncorrected ASR')+' target speech concatenated into one paragraph per recording; no source turn or row cuts'
        policy.update(segmentation='Pretrained left/right word-gap model with rule features',
                      automatic_boundary_rule='Paragraph-derived rules are joint model features; source row boundaries are not utterance cuts')

    report = {
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_type': dataset_type,
        'metric_profile': metric_profile,
        'boundary_context':boundary_context,
        'method': method_name,
        'segmenter': evidence,
        'scoring_policy': policy,
        'scope': scope_desc,
        'versions': {n: version(n) for n in ('spacy', 'en-core-web-sm', 'taaled', 'pylats')},
        'code_sha256': {p: digest(ROOT / p) for p in SOURCE_FILES+(("speech_pipeline/segmentation.py",) if trained else ())},
        'recordings': [],
        'patients': [],
    }

    pooled = defaultdict(list)
    with TemporaryDirectory() as temp:
        for (patient, filename), speaker in sorted(mapping.items()):
            if is_corr:
                c_filename = get_corrected_filename(patient, filename)
                path = ROOT / 'dataset/corrected' / patient / c_filename
                lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                if source_lines:
                    utterances = [{'utterance_id': i + 1, 'text': line} for i, line in enumerate(lines)]
                    turn_mapping = []
                else:
                    from speech_pipeline.clan import predict_corrected,corrected_turns
                    raw = ROOT / 'dataset/uncorrected' / patient / filename
                    if boundary_context=='paragraph':
                        paragraph=' '.join(auto.WORD.findall(path.read_text(encoding='utf-8')))
                        utterances=trained.segment(paragraph);turn_mapping=[]
                        for u in utterances:
                            u['paragraph_char_start']=u.pop('source_start');u['paragraph_char_end']=u.pop('source_end')
                            u['context']='one_target_speaker_paragraph'
                    elif trained:
                        _,turns,turn_mapping,_=corrected_turns(list(auto.read_rows(raw)),speaker,path.read_text(encoding='utf-8'))
                        utterances=[]
                        for turn in turns:
                            for u in trained.segment(turn['text']):
                                start,end=u.pop('source_start'),u.pop('source_end')
                                u.update(utterance_id=len(utterances)+1,turn_id=turn['turn_id'],source_rows=turn['source_rows'],
                                    corrected_char_start=turn['corrected_char_start']+start,
                                    corrected_char_end=turn['corrected_char_start']+end,
                                    turn_mapping_review_required=turn['turn_mapping_review_required'])
                                utterances.append(u)
                    else:
                        _, utterances, turn_mapping, _ = predict_corrected(list(auto.read_rows(raw)), speaker,
                            path.read_text(encoding='utf-8'), nlp)
                source_rows = len(lines)
                source_display = c_filename
                key = f'{patient}_{Path(c_filename).stem}_{speaker}'
            else:
                path = ROOT / 'dataset/uncorrected' / patient / filename
                rows = list(auto.read_rows(path))
                utterances = (trained.target_paragraph(rows,speaker) if boundary_context=='paragraph' else
                              trained.target_rows(rows,speaker)) if trained else auto.predict(rows,speaker,nlp)
                original = '\n'.join(r['text'] for r in rows if r['speaker'].casefold() == speaker.casefold() and r['text'])
                source_rows = len(original.splitlines())
                source_display = filename
                key = f'{patient}_{Path(filename).stem}_{speaker}'

            if not utterances:
                raise ValueError(f'No target speech: {path}')

            body = auto.chat_text(utterances, f'{patient}/{source_display}; speaker {speaker}')
            sha = hashlib.sha256(body.encode()).hexdigest()
            chat = out_dir / 'clan_software_input_files' / f'{key}_auto_{sha[:12]}.cha'
            save(out_dir / 'segmented_utterances' / f'{key}.json', {
                'method': method_name,
                'raw_sha256': digest(path),
                'source_file': f'{patient}/{source_display}',
                'target_speaker': speaker,
                'utterances': utterances,
                **({'turn_mapping': turn_mapping} if is_corr else {}),
                **({'paragraph_input_policy':'Original punctuation, source row and speaker-turn cuts withheld; spelling retained. Offsets refer to reconstructed paragraph.'} if boundary_context=='paragraph' else {})
            })

            record = {
                'patient_id': patient,
                'source_file': source_display,
                'target_speaker': speaker,
                'raw_sha256': digest(path),
                'source_rows': source_rows,
                'utterance_file': f'segmented_utterances/{key}.json',
                'prediction_file': f'segmented_utterances/{key}.json',
                'chat_file': str(chat.relative_to(out_dir)),
                'chat_sha256': sha,
                'mlu': auto.word_mlu(utterances),
                'lexical': lexical(utterances, nlp, Path(temp) / 'sample.txt'),
                'clan': {'status': 'no_matching_saved_reference; rerun_CLAN_required'}
            }
            report['recordings'].append(record)
            if verify_clan:record['clan']=clan_software_check(utterances,Path(temp))
            if metric_profile == 'clinical':
                record['requested_metrics'] = custom.clinical_from_predictions(utterances)
            if is_corr and boundary_context=='speaker-turns':
                record['turn_mapping_review_required'] = sum(a['review_required'] for a in turn_mapping)
            elif is_corr:record['turn_mapping_status']='not_applied_paragraph_mode'
            pooled[patient].extend(utterances)
            print(f'[{dataset_type}] {len(report["recordings"])}/{len(mapping)} {patient}: {len(utterances)} utterances', flush=True)

        for patient, utterances in pooled.items():
            report['patients'].append({
                'patient_id': patient,
                'pool_order': 'source filename ascending',
                'mlu': auto.word_mlu(utterances),
                'lexical': lexical(utterances, nlp, Path(temp) / 'pooled.txt')
            })
            if metric_profile == 'clinical':
                report['patients'][-1]['requested_metrics'] = custom.clinical_from_predictions(utterances)
            print(f'[{dataset_type}] Pooled {patient}', flush=True)

    save(out_dir / 'comparison.json', report)
    export_all(report, out_dir)

    (out_dir / 'clan_software_input_files').mkdir(parents=True, exist_ok=True)
    for r in report['recordings']:
        path_key = r.get('utterance_file') or r.get('prediction_file')
        data = json.loads((out_dir / path_key).read_text())
        body = auto.chat_text(data['utterances'], f"{r['patient_id']}/{r['source_file']}; speaker {r['target_speaker']}")
        assert hashlib.sha256(body.encode()).hexdigest() == r['chat_sha256']
        (out_dir / r['chat_file']).write_text(body)

    current = {Path(r['chat_file']).name for r in report['recordings']}
    for p in (out_dir / 'clan_software_input_files').glob('*_auto_*.cha'):
        if p.name not in current:
            p.unlink()

    print(f'[{dataset_type}] Saved outputs to {out_dir}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clan-results', type=Path)
    parser.add_argument('--metric-profile', choices=['standard', 'clinical'], default='standard',
                        help='Clinical profile exports the requested surface measures and explicit missing-annotation statuses')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--corrected-boundary-mode', choices=['automatic', 'source-lines'], default='automatic',
                        help='Corrected segmentation; source-lines reproduces the earlier reference procedure')
    parser.add_argument('--dataset', choices=['uncorrected', 'corrected', 'all'], default='all',
                        help='Dataset to process: uncorrected, corrected, or all (default: all)')
    parser.add_argument('--segmenter',choices=['auto','rules','improved'],default='auto',
                        help='auto uses the trained model for plain paragraphs and established rules for workbooks; improved explicitly enables experimental workbook transfer')
    parser.add_argument('--paragraph',type=Path,help='Analyze a plain-text paragraph instead of the workbook datasets')
    parser.add_argument('--boundary-context',choices=['speaker-turns','paragraph'],default='speaker-turns',
                        help='For workbooks, paragraph joins target speech per recording before model segmentation; speaker-turns retains the existing input context')
    parser.add_argument('--verify-clan',action='store_true',help='Run installed official CLAN MLU/FREQ on the exact exported words')
    args = parser.parse_args()
    if args.paragraph and args.clan_results:parser.error('--paragraph and --clan-results cannot be combined')

    if args.clan_results:
        target_datasets = [args.dataset] if args.dataset != 'all' else ['uncorrected', 'corrected']
        for ds in target_datasets:
            target_out = args.output_root / ds
            comp_path = target_out / 'comparison.json'
            if not comp_path.exists():
                continue
            report = json.loads(comp_path.read_text())
            imported = 0
            for record in report['recordings']:
                chat = target_out / record['chat_file']
                if digest(chat) != record['chat_sha256']:
                    raise ValueError(f"CHAT changed since prediction; regenerate before comparison: {chat}")
                ref = args.clan_results / (chat.stem + '.mlu.cex')
                if ref.exists():
                    record['clan'] = clan_reference(ref, chat.stem, record['mlu'])
                    imported += 1
            if imported:
                save(comp_path, report)
                export_all(report, target_out)
                print(f'[{ds}] Imported {imported}/{len(report["recordings"])} CLAN results')
        return

    nlp = spacy.load('en_core_web_sm')
    if args.paragraph:
        run_paragraph(args.paragraph,nlp,args.output_root,args.metric_profile,args.segmenter,args.verify_clan)
        return
    if args.dataset in ('uncorrected', 'all'):
        print("=== Processing UNCORRECTED transcripts ===")
        run_pipeline('uncorrected', nlp, args.output_root, metric_profile=args.metric_profile,segmenter=args.segmenter,verify_clan=args.verify_clan,boundary_context=args.boundary_context)
    if args.dataset in ('corrected', 'all'):
        print("=== Processing CORRECTED transcripts ===")
        run_pipeline('corrected', nlp, args.output_root, args.corrected_boundary_mode, args.metric_profile,args.segmenter,args.verify_clan,args.boundary_context)


if __name__ == '__main__':
    main()
