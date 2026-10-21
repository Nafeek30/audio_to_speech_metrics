"""Acquire CHAT corpora and evaluate text segmentation against published labels.

Downloads respect TalkBank authentication. No login page is treated as data.
Gold CHAT punctuation, timestamps and line breaks are excluded from predictor
inputs; speaker changes remain available. This is transcript-only validation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import zipfile

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.pipeline import digest, save, lexical

CORPORA = {
    'nyu_emerson': dict(page='https://talkbank.org/asd/access/English/NYU-Emerson.html',
        transcripts='https://talkbank.org/data/asd/English/NYU-Emerson?f=zip',
        media='https://media.talkbank.org/asd/English/NYU-Emerson',
        citation='Steen et al. (2026), Introducing the NYU-Emerson Corpus, Meeting on Language in Autism.',
        doi='10.21415/VT7P-YF83'),
    'ocsc': dict(page='https://talkbank.org/childes/access/Eng-NA/OCSC.html',
        transcripts='https://talkbank.org/data/childes/Eng-NA/OCSC?f=zip',
        media='https://media.talkbank.org/childes/Eng-NA/OCSC',
        citation='Wagner et al. (2025), The Ohio Child Speech Corpus, Speech Communication 170, 103206.',
        doi='10.21415/60TV-D669'),
}
DATA = ROOT / 'dataset/public'
OUT = ROOT / 'outputs/results/benchmark'
EVIDENCE = ROOT / 'outputs/validation/benchmark'
DISCOURSE_GEMS = {'Howto', 'DescriptivePictures', 'ExpPictures'}


def normalize_gem(value):
    """Normalize observed header spelling/punctuation, retaining the raw header."""
    cleaned=re.sub(r'[\s.=]+$', '', value.strip())
    known={v.casefold():v for v in (*DISCOURSE_GEMS,'Alphabet','Numbers','Wug','Reading','EndTasks')}
    return known.get(cleaned.casefold(),cleaned)


def install_transcripts(name, archive):
    """Extract an authorized local ZIP safely, without retaining a duplicate ZIP."""
    if name not in CORPORA:
        raise ValueError(f'Unknown corpus: {name}')
    archive = Path(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError('Expected a transcript ZIP; an HTML login page is not a dataset')
    target = DATA / name / 'transcripts'
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir() and '__MACOSX' not in m.filename and not m.filename.endswith('.DS_Store')]
        if not any(m.filename.lower().endswith('.cha') for m in members):
            raise ValueError('Archive contains no CHAT transcripts')
        for m in members:
            p = (target / m.filename).resolve()
            if not p.is_relative_to(target.resolve()) or (m.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe path or symlink in transcript archive')
        for m in members:
            p = target / m.filename
            if p.exists() and p.read_bytes() != z.read(m):
                raise FileExistsError(f'Existing transcript differs; preserve and review before replacing: {p}')
        for m in members:
            p = target / m.filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(m))
    save(DATA / name / 'source.json', dict(**CORPORA[name], archive_sha256=digest(archive),
        acquired_utc=datetime.now(timezone.utc).isoformat(),
        chat_files=len(list(target.rglob('*.cha')))))
    return target


def fetch(name):
    with TemporaryDirectory(prefix='talkbank_download_') as temp:
        path = Path(temp) / 'transcripts.zip'
        result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
            '--max-time', '180', CORPORA[name]['transcripts'], '--output', str(path)], capture_output=True, text=True)
        if result.returncode:
            return dict(corpus=name, status='download_failed', detail=result.stderr.strip())
        if not zipfile.is_zipfile(path):
            body = path.read_text(errors='replace')[:10000]
            gated = any(x in body.lower() for x in ('authmodals', 'login', 'sign in'))
            return dict(corpus=name, status='login_required' if gated else 'unexpected_download_content',
                        detail='Sign in to TalkBank and download the transcript ZIP; then use import-zip. Media access also requires login.')
        folder = install_transcripts(name, path)
        return dict(corpus=name, status='downloaded', path=str(folder.relative_to(ROOT)))


def read_chat(path):
    """Read faithful spoken text, keeping annotation evidence separate."""
    try:
        import pylangacq
    except ImportError as e:
        raise RuntimeError('Install the public-data extra: uv sync --extra figures --extra public-data') from e
    reader = pylangacq.read_chat(str(path), strict=True)
    # Initial @G/@BG headers are stored among file headers rather than returned
    # as changeable utterances by some readers. Align gems to raw main tiers.
    raw_gems = []
    raw_gem = 'Unspecified'
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line.startswith(('@G:', '@BG:')):
            raw_gem = line.split(':', 1)[1].strip()
        elif line.startswith('@EG:'):
            raw_gem = 'Unspecified'
        elif line.startswith('*'):
            raw_gems.append(raw_gem)
    gem = 'Unspecified'
    rows = []
    for u in reader.utterances():
        ch = u.changeable_header
        if ch is not None:
            if isinstance(ch, (pylangacq.ChangeableHeader.G, pylangacq.ChangeableHeader.Bg)):
                gem = ch.value
            elif isinstance(ch, pylangacq.ChangeableHeader.Eg):
                gem = 'Unspecified'
            continue
        raw = u.tiers[u.participant]
        if len(rows) >= len(raw_gems):
            raise ValueError('CHAT main-tier/gem alignment could not be verified')
        raw_gem = raw_gems[len(rows)]
        gem = normalize_gem(raw_gem)
        text = u.audible or ''
        # Tokenize before joining. Never pass published utterance punctuation to
        # spaCy or the learned model as a hidden answer key.
        # ``audible`` retains &= non-speech event codes (e.g. &=laughs).
        # Their descriptions are not spoken words. Strip the entire event,
        # while preserving actual fillers, repetitions and word fragments.
        spoken=re.sub(r'&=[^\s<>\[\]]+',' ',text)
        input_text = ' '.join(auto.WORD.findall(spoken))
        rows.append(dict(row=len(rows)+2, speaker=u.participant, text=input_text,
            audible=text, raw_chat=raw, gem=gem, raw_gem=raw_gem,
            start='', end='', time_marks=list(u.time_marks) if u.time_marks else None,
            parsed_main_body=[t.word for t in u.tokens if t.word and auto.WORD.fullmatch(t.word)],
            morphology_present=bool(u.tiers.get('%mor')),
            grammar_error_marks=len(re.findall(r'\[\+\s*gram\s*\]', raw)),
            omission_marks=len(re.findall(r'(?<!\w)0(?:[A-Za-z]|\S+)', raw)),
            substitution_marks=len(re.findall(r'\[:[^\]]+\]', raw)),
            retrace_marks=len(re.findall(r'\[/{1,3}\]', raw)),
            filled_pause_marks=len(re.findall(r'&-[^\s]+', raw))))
    if len(rows) != len(raw_gems):
        raise ValueError('CHAT main-tier/gem alignment could not be verified')
    return rows


def chat_reference_units(rows,speaker='CHI'):
    """Reference-only CHAT continuation handling; never predictor features.

    CLAN joins an incompletion followed by +, for the same speaker, and
    transcription breaks +. continue into the next same-speaker segment.
    Non-speech-only rows do not become verbal utterances.
    """
    units=[];incomplete=False;forced=False
    for row in rows:
        if row['speaker']!=speaker or not row['text']:continue
        syntax=re.sub(r'\[[^\]]*\]|\x15[^\x15]*\x15',' ',row['raw_chat'])
        completion=bool(re.search(r'(?:^|\s)\+,(?=\s|$)',syntax))
        if units and (forced or incomplete and completion):
            units[-1]['text']+=' '+row['text'];units[-1]['source_rows'].append(row['row'])
        else:units.append(dict(text=row['text'],source_rows=[row['row']]))
        incomplete=bool(re.search(r'(?:^|\s)\+(?:/{1,2}[.?]|\.{3}|\.\.\?|…)(?=\s|[>\[]|$)',syntax))
        forced=bool(re.search(r'(?:^|\s)\+\.(?=\s|[>\[]|$)',syntax))
    return units


def media_inventory(folder):
    files = [p for p in folder.rglob('*') if p.suffix.lower() in {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.mov'}]
    return {p.stem: p for p in files}


def inspect_corpus(name):
    base = DATA / name
    media = media_inventory(base)
    records = []
    for path in sorted((base/'transcripts').rglob('*.cha')):
        try:
            rows = read_chat(path)
            text = path.read_text(encoding='utf-8-sig')
            m = re.search(r'^@Media:\s*([^,\n]+)', text, re.M)
            media_id = m[1].strip() if m else path.stem
            records.append(dict(corpus=name, source=str(path.relative_to(base)), status='parsed',
                source_sha256=digest(path), speakers=sorted({r['speaker'] for r in rows}),
                gems=sorted({r['gem'] for r in rows}), utterances=len(rows),
                morphology_utterances=sum(r['morphology_present'] for r in rows),
                timed_utterances=sum(r['time_marks'] is not None for r in rows),
                media_id=media_id, media_available=media_id in media,
                annotation_marks={k:sum(r[k] for r in rows) for k in
                    ('grammar_error_marks','omission_marks','substitution_marks','retrace_marks','filled_pause_marks')}))
        except (ValueError, OSError) as e:
            records.append(dict(corpus=name, source=str(path.relative_to(base)), status='parse_error', detail=str(e)))
    return records


def prepare_training(names, participant_map, speaker='CHI', gems=None, nlp=None):
    """Use published CHAT boundaries as training targets, never as input features.

    The reviewed identity table joins repeated sessions of the same child. A
    CHAT speaker code such as CHI is only local to a recording, not a person ID.
    """
    if nlp is None:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    with Path(participant_map).open(encoding='utf-8-sig', newline='') as f:
        identities = list(csv.DictReader(f))
    mapping = {}
    for row in identities:
        key = (row['corpus'], row['recording'], row['speaker'])
        if key in mapping:
            raise ValueError(f'Duplicate participant mapping: {key}')
        if row.get('verified', '').strip().lower() not in {'true', '1', 'yes'} or not row['participant'].strip():
            raise ValueError(f'Participant mapping must be verified: {key}')
        mapping[key] = row['participant'].strip()
    records, sources = [], {}
    for name in names:
        if name not in CORPORA:
            raise ValueError(f'Unknown corpus: {name}')
        paths = sorted((DATA/name/'transcripts').rglob('*.cha'))
        if not paths:
            raise FileNotFoundError(f'Import the authorized {name} transcript ZIP first')
        selected_gems=set(gems) if gems else (DISCOURSE_GEMS if name=='ocsc' else None)
        before=len(records)
        for path in paths:
            relative = path.relative_to(DATA/name).as_posix()
            rows = read_chat(path)
            if not any(r['speaker'] == speaker and r['text'] for r in rows):
                continue
            key = (name, relative, speaker)
            if key not in mapping:
                raise ValueError(f'Missing reviewed participant identity for {key}')
            sources[f'{name}/{relative}'] = digest(path)
            # Keep contiguous task sections separate. Never concatenate different
            # tasks or use a gold utterance break to terminate a model context.
            sections = []
            for row in rows:
                if not sections or sections[-1][0]['gem'] != row['gem']:
                    sections.append([])
                sections[-1].append(row)
            for section_index, section in enumerate(sections):
                if selected_gems and section[0]['gem'] not in selected_gems:
                    continue
                gold = chat_reference_units(section,speaker)
                tokens, boundaries = auto.stream(gold)
                if not tokens:continue
                # The predictor sees one continuous child-speech paragraph.
                # Adult turns and published row boundaries supply no cuts.
                paragraph=' '.join(tokens)
                predicted=auto.predict([dict(row=2,speaker=speaker,text=paragraph,start='',end='')],speaker,nlp)
                predicted_tokens, rule_boundaries=auto.stream(predicted)
                if tokens!=predicted_tokens:
                    raise ValueError(f'Rule segmentation changed the word stream in {relative}')
                surface=[w for u in gold for w in auto.WORD.findall(u['text'])]
                baseline=[int(i+1 in rule_boundaries) for i in range(len(tokens))]
                labels=[int(i+1 in boundaries) if i+1<len(tokens) and
                    not({tokens[i],tokens[i+1]}&{'xxx','yyy','www'}) else -100 for i in range(len(tokens))]
                if not any(v >= 0 for v in labels):
                    continue
                if len(surface) != len(tokens):
                    raise ValueError(f'Surface-word alignment failed: {relative}')
                records.append(dict(corpus=name, participant=f'{name}/{mapping[key]}', participant_verified=True,
                    task='paragraph_utterance_boundaries',
                    recording=f'{relative}#section{section_index+1}', gem=section[0]['gem'],
                    tokens=tokens, surface_words=surface, baseline=baseline, labels=labels,
                    gold_boundaries=sorted(boundaries), source_sha256=digest(path),
                    input_policy='One continuous child-speech paragraph; case-folded words. Gold punctuation, rows, speaker turns, CHAT codes, morphology and timestamps withheld. Rule predictions are stored separately; the trainer selects words-only or joint features.',
                    label_policy='CHAT utterance ends as targets; paragraph-final and unintelligible-adjacent gaps masked'))
        if len(records)==before:
            raise ValueError(f'No usable labeled speech in selected {name} tasks; inspect its task inventory')
    if not records:
        raise ValueError('No usable labeled speech in the requested speakers/tasks')
    path = EVIDENCE/'talkbank_training.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    save(EVIDENCE/'talkbank_training_sources.json', dict(sources=sources,
        participant_map_sha256=digest(participant_map), dataset_sha256=digest(path),
        records=len(records), participants=len({r['participant'] for r in records}),
        corpora=names, speaker=speaker, gems=gems or {'ocsc':sorted(DISCOURSE_GEMS),'nyu_emerson':'all'},
        task='Predict utterance ends from a continuous unpunctuated paragraph; words only, no rule features or speaker-turn cuts',
        limitation='Child speech is concatenated within each task; intervening adult speech is omitted. No ASR noise, grammaticality or maze labels are fabricated',
        by_corpus={name:dict(records=sum(r['corpus']==name for r in records),
            participants=len({r['participant'] for r in records if r['corpus']==name}),
            words=sum(len(r['tokens']) for r in records if r['corpus']==name),
            labeled_gaps=sum(v>=0 for r in records if r['corpus']==name for v in r['labels'])) for name in names}))
    return dict(path=str(path), records=len(records), participants=len({r['participant'] for r in records}),
                labeled_gaps=sum(v >= 0 for r in records for v in r['labels']))


def run_clan(tool, source, speaker='PAR', *, morphology=False, window=None):
    """CLAN requires terminal stdin/stdout; launch in a bounded pseudo-terminal."""
    binary = ROOT / '.venv/bin' / f'clan-{tool}'
    if not binary.exists():
        binary = Path(shutil.which(tool) or '')
    if not binary.is_file():
        raise FileNotFoundError(f'Official CLAN {tool} executable is unavailable')
    with TemporaryDirectory(prefix='clan_reference_') as temp:
        chat = Path(temp)/'sample.cha'
        shutil.copy2(source, chat)
        args = [str(binary), f'+t*{speaker}', '+t%mor' if morphology else '-t%mor', '+f']
        if tool == 'freq':
            args.append('+k')  # FREQ is case-sensitive by default; match case folding.
        if window:
            args.append(f'+b{window}')
        args.append(str(chat))
        master, slave = pty.openpty()
        try:
            # Absolute input paths make CLAN write beside the temporary source.
            # Omitting cwd permits posix_spawn on macOS, avoiding an expensive
            # fork of the training process and its large model allocator.
            # Python-created descriptors are non-inheritable; the PTY is duped
            # explicitly to standard streams by Popen.
            p = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave,
                                 close_fds=False)
            os.close(slave);slave=None
            end = time.monotonic()+90
            while p.poll() is None:
                if time.monotonic()>end:
                    p.kill();p.wait();raise TimeoutError(f'CLAN {tool} timed out')
                if select.select([master], [], [], .1)[0]:
                    try: os.read(master, 65536)
                    except OSError: break
            p.wait(timeout=5)
            if p.returncode:
                raise RuntimeError(f'CLAN {tool} exited with {p.returncode}')
        finally:
            os.close(master)
            if slave is not None:os.close(slave)
        outputs=list(Path(temp).glob('*.cex'))
        if len(outputs)!=1:
            raise RuntimeError(f'CLAN {tool} produced {len(outputs)} output files')
        text=outputs[0].read_text(errors='replace')
    result={'command':f'{tool} +t*{speaker} '+('+t%mor' if morphology else '-t%mor')+(' +k' if tool=='freq' else '')+(f' +b{window}' if window else ''),
            'output_sha256':hashlib.sha256(text.encode()).hexdigest(), 'output':text}
    def number(pattern):
        m=re.search(pattern,text,re.I)
        return float(m[1]) if m else None
    if tool=='mlu':
        result.update(utterances=number(r'utterances\s*=\s*(\d+)'),
                      total=number(r'(?:morphemes|words)\s*=\s*(\d+)'),
                      mlu=number(r'Ratio of (?:morphemes|words) over utterances\s*=\s*([\d.]+)'),
                      sd=number(r'Standard deviation\s*=\s*([\d.]+)'))
    elif tool=='freq':
        result.update(NDW=number(r'(\d+)\s+Total number of different item types'),
                      TNW=number(r'(\d+)\s+Total number of items \(tokens\)'),
                      TTR=number(r'([\d.]+)\s+Type/Token ratio'),
                      MATTR=number(r'([\d.]+)\s+MATTR'))
    return result


def calculator_checks(utterances, nlp, temp, label):
    """Compare calculators on identical explicitly normalized words, not gold coding."""
    from speech_pipeline.metrics import compute_mattr
    normalized = []
    for u in utterances:
        words=auto.score_utterance(u['text'])['counted_words']
        if words:normalized.append(dict(text=' '.join(words),utterance_id=len(normalized)+1))
    if not normalized:return None
    path=temp/f'{label}.cha';path.write_text(auto.chat_text(normalized,label))
    words=[w.casefold() for u in normalized for w in auto.WORD.findall(u['text'])]
    mlu=run_clan('mlu',path);freq=run_clan('freq',path,window=50 if len(words)>=50 else None)
    custom=dict(TNW=len(words),NDW=len(set(words)),TTR=len(set(words))/len(words),
                MATTR=compute_mattr(words,50)[0] if len(words)>=50 else None,
                MLUw=len(words)/len(normalized))
    official={k:freq[k] for k in ('TNW','NDW','TTR','MATTR')};official['MLUw']=mlu['mlu']
    checks={k:dict(custom=custom[k],clan=official[k],
                  absolute_error=abs(custom[k]-official[k]) if custom[k] is not None and official[k] is not None else None)
            for k in custom}
    taaled=lexical(normalized,nlp,temp/'lexical.txt')
    from speech_pipeline import metrics as custom, taaled as library
    text='\n'.join(auto.analysis_text(u['text']) for u in normalized)
    tokens,tagged,metadata=custom.preprocess_text(text,nlp)
    mine=custom.calculate_speech_metrics(text,tokens,tagged,metadata)
    scores=library.ld.lexdiv(tokens,window_length=50,samples=min(42,len(tokens)))
    token_reference=dict(basic_ntokens=scores.ntokens,basic_ntypes=scores.ntypes,
                         simple_ttr_aw=scores.ttr,mattr50_aw=scores.mattr if len(tokens)>=50 else None)
    same_tokens={k:dict(custom=mine[k],taaled=v,absolute_error=abs(mine[k]-v) if v is not None else None)
                 for k,v in token_reference.items()}
    return dict(surface_calculator_checks=checks,taaled_normalized_checks=taaled,
        taaled_same_tokens_checks=same_tokens,
        policy='Case-folded filler-excluded surface text for CLAN/custom. TAALED same-token checks isolate calculators; normalized checks include differences between the two preprocessing implementations.',
        official_outputs={'mlu':mlu,'freq':freq})


def benchmark(name, speaker='CHI', gems=None):
    import spacy
    nlp=spacy.load('en_core_web_sm')
    records=[]
    source_references=[]
    corpus=DATA/name
    for path in sorted((corpus/'transcripts').rglob('*.cha')):
        try:
            rows=read_chat(path)
            if speaker not in {r['speaker'] for r in rows}:
                raise ValueError(f'Target speaker {speaker} is absent')
            if not any(r['text'] for r in rows if r['speaker']==speaker):
                records.append(dict(corpus=name,recording=str(path.relative_to(corpus)),
                                    status='no_usable_target_speech'))
                continue
            reference=dict(recording=str(path.relative_to(corpus)),
                scope='Full original CHAT file, all tasks; distinct from the per-task stripped-text benchmark',
                MLUw=run_clan('mlu',path,speaker),
                lexical=run_clan('freq',path,speaker,window=50))
            if all(r['morphology_present'] for r in rows if r['speaker']==speaker and r['text']):
                reference['MLUm']=run_clan('mlu',path,speaker,morphology=True)
            source_references.append(reference)
            for gem in sorted({r['gem'] for r in rows}):
                if gems and gem not in gems:continue
                section=[]
                for r in [r for r in rows if r['gem']==gem]:
                    if section and r['row']!=section[-1]['row']+1:
                        section.append(dict(row=r['row']-1,speaker='_SECTION_BREAK_',text='section boundary',start='',end=''))
                    section.append(r)
                gold=[dict(text=r['text']) for r in section if r['speaker']==speaker and r['text']]
                if not gold:continue
                predicted=auto.predict(section,speaker,nlp)
                c=auto.boundary_counts(gold,predicted)
                truth=auto.word_mlu(gold);mine=auto.word_mlu(predicted)
                with TemporaryDirectory(prefix='corpus_benchmark_') as temp:
                    calculators=calculator_checks(gold,nlp,Path(temp),'same_text')
                records.append(dict(corpus=name,recording=str(path.relative_to(corpus)),gem=gem,
                    source_sha256=digest(path),status='completed_transcript_only',
                    input_policy='Published words and speaker turns; CHAT punctuation, morphology, timestamps, and utterance breaks withheld from predictor',
                    reference_metrics=truth,pipeline_metrics=mine,counts=c,**auto.boundary_scores(c),
                    calculator_checks=calculators,
                    additional_metrics={m:None for m in ('MLUm','WPM','PGU','omission_count','substitution_count','maze_percentage')},
                    missing_metrics_reason='Require a separately validated morphology/timing/grammar/maze prediction and matched annotation protocol; gold labels are not pipeline predictions'))
        except (ValueError,OSError,RuntimeError,TimeoutError) as e:
            records.append(dict(corpus=name,recording=str(path.relative_to(corpus)),status='failed',detail=str(e)))
    if not records:
        raise FileNotFoundError(f'No CHAT files in {corpus}/transcripts; authenticate/download and import them first')
    save(OUT/f'{name}.json',dict(records=records,original_chat_software_references=source_references,
        scope='Text segmentation and same-input calculator validation; not audio ASR or clinical validation',
        source=CORPORA[name]))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    training=commands.add_parser('prepare-training')
    training.add_argument('corpora', choices=CORPORA, nargs='+')
    training.add_argument('--participant-map', type=Path, required=True)
    training.add_argument('--speaker', default='CHI')
    training.add_argument('--gems', nargs='+')
    for name in ('download','inspect','benchmark','import-zip'):
        sub=commands.add_parser(name);sub.add_argument('corpus',choices=CORPORA)
        if name=='import-zip':sub.add_argument('archive',type=Path)
        if name=='benchmark':
            sub.add_argument('--speaker',default='CHI');sub.add_argument('--gems',nargs='+')
    args=parser.parse_args()
    if args.command=='prepare-training':
        result=prepare_training(args.corpora,args.participant_map,args.speaker,args.gems)
    elif args.command=='download':
        result=fetch(args.corpus)
        path=EVIDENCE/'access.json';previous=json.loads(path.read_text()) if path.exists() else {}
        previous[args.corpus]=dict(**CORPORA[args.corpus],**result);save(path,previous)
    elif args.command=='import-zip':
        folder=install_transcripts(args.corpus,args.archive)
        result=dict(corpus=args.corpus,status='transcripts_imported',path=str(folder.relative_to(ROOT)))
        path=EVIDENCE/'access.json';previous=json.loads(path.read_text()) if path.exists() else {}
        previous[args.corpus]=dict(**CORPORA[args.corpus],**result);save(path,previous)
    elif args.command=='inspect':
        result=inspect_corpus(args.corpus);save(EVIDENCE/f'{args.corpus}_inventory.json',result)
    else:result=benchmark(args.corpus,args.speaker,args.gems)
    print(json.dumps(result,indent=2))
    if args.command=='download' and result['status']!='downloaded':
        raise SystemExit(2)


if __name__=='__main__':main()
