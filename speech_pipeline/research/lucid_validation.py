"""Compare frozen lexical implementations with actual published LUCID results.

This is oracle-transcript lexical validation, not MOSS ASR or CHAT-unit gold.
Register cleaning/mapping before scoring. Ambiguous recording matches remain
missing; never pick the file whose numbers best match the publication.
"""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from speech_pipeline import ROOT

DATA=ROOT/'dataset/public/lucid_published'
OUT=ROOT/'outputs/results/lucid_validation'
EVIDENCE=ROOT/'outputs/validation/lucid_validation'
SHARED=('basic_ntokens','basic_ntypes','basic_ncontent_tokens','basic_ncontent_types',
        'basic_nfunction_tokens','basic_nfunction_types','lexical_density_tokens','lexical_density_types',
        'simple_ttr_aw','log_ttr_aw','maas_ttr_aw','mattr50_aw','msttr50_aw','hdd42_aw','mtld_original_aw')
NAME=re.compile(r'^(?:LCD_)?(?P<speaker>[FM]\d+)(?P<partner>(?:BE[FM]|C[FM]|[FM])\d+)(?P<scene>[BFS])(?P<image>\d+)(?P<condition>clL2|cln|clv|cv)(?P<position>\d+)(?:_Nuc)?$')
INTERVAL=re.compile(r'(?m)^\s*(\d+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\n"((?:""|[^"])*)"')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def file_metadata(path):
    match=NAME.fullmatch(Path(path).stem)
    if not match:raise ValueError('Unrecognized documented LUCID file name')
    d=match.groupdict();d['condition']={'cv':'noBarrier','clv':'vocoder','cln':'babble','clL2':'L2'}[d['condition']]
    d['scene']={'B':'beach','F':'farm','S':'shop'}[d['scene']]
    return d


def parse_chronological(text):
    """Strictly read the released chronological Praat interval-tier format."""
    lines=text.splitlines()
    if not lines or lines[0].strip()!='"Praat chronological TextGrid text file"':raise ValueError('Unsupported TextGrid format')
    tier_count=int(lines[2].split('!')[0].strip())
    tiers=[]
    for line in lines[3:3+tier_count]:
        m=re.fullmatch(r'\s*"IntervalTier"\s+"([^"]*)"\s+[-+\d.eE]+\s+[-+\d.eE]+\s*',line)
        if not m:raise ValueError('Unsupported TextGrid tier header')
        tiers.append(m[1])
    body='\n'.join(lines[3+tier_count:]);rows=[];previous=0
    def padding(s):return not '\n'.join(x for x in s.splitlines() if not x.lstrip().startswith('!')).strip()
    for m in INTERVAL.finditer(body):
        if not padding(body[previous:m.start()]):raise ValueError('Unparsed TextGrid content')
        tier=int(m[1]);start=float(m[2]);end=float(m[3]);label=m[4].replace('""','"')
        if not 1<=tier<=len(tiers) or not np.isfinite([start,end]).all() or start<0 or end<start:
            raise ValueError('Invalid TextGrid interval')
        rows.append(dict(tier=tiers[tier-1],start=start,end=end,label=label));previous=m.end()
    if not padding(body[previous:]):raise ValueError('Unparsed trailing TextGrid content')
    if not rows:raise ValueError('No TextGrid intervals')
    return tiers,rows


def cleaned_labels(labels):
    """Remove explicit non-speech codes, retaining spoken lexical payloads.

    Existing production filler/fragment/token/POS policies apply afterward.
    No policy is inferred from the published numeric scores.
    """
    words=[];removed=Counter();payloads=Counter()
    for label in labels:
        value=label.replace('\x7f','').strip()
        if not value:removed['empty']+=1;continue
        if re.fullmatch(r'<?!?SIL>?',value,re.I):removed['silence']+=1;continue
        tag=re.fullmatch(r'<([A-Za-z!]+)(?:_([^>]*))?>?',value)
        if tag:
            if tag[2]:value=tag[2].replace('_',' ');payloads[tag[1]]+=1
            else:removed[tag[1]]+=1;continue
        # Source-only annotation repair, registered before any scoring. The
        # released grids sometimes have >LG>, <LS<, LG_word>, embedded <BR>,
        # or a spare > after an otherwise ordinary word. Never edit raw files.
        codes=r'BR|LS|GA|LG|LA|!?SIL'
        def embedded(match):
            code,payload=match.group(1),match.group(2)
            if payload:payloads[code]+=1;return payload.replace('_',' ')
            removed[code]+=1;return ''
        value=re.sub(rf'[<>]?({codes})(?:_([^<>]*))?[<>]',embedded,value)
        value=re.sub(rf'[<>]({codes})[.,]?(?=\s|$)',lambda m:removed.update([m[1]]) or '',value)
        split=re.fullmatch(r'<([A-Z])>([A-Z])',value)
        if split and split[1]+split[2] in {'BR','LS','GA','LG','LA'}:
            removed[split[1]+split[2]]+=1;continue
        if '<' in value or '>' in value:removed['stray_angle_delimiter']+=1;value=value.replace('<','').replace('>','')
        value=value.strip()
        if not value:continue
        if any(c in value for c in '<>[]@&+*=|\\'):
            raise ValueError(f'Unresolved source annotation: {value!r}')
        words.append(value)
    return ' '.join(words),dict(removed),dict(payloads)


def references():
    import csv
    with (DATA/'TAALED_results.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def prepare():
    if (EVIDENCE/'protocol.json').exists():raise FileExistsError('Preserve registered lexical comparison')
    index=defaultdict(list);unrecognized=[];sources=[]
    for path in sorted((DATA/'textgrids').rglob('*.TextGrid')):
        try:m=file_metadata(path)
        except ValueError as e:unrecognized.append(dict(file=str(path.relative_to(ROOT)),reason=str(e)));continue
        k=m['speaker'],m['condition'],m['scene'];index[k].append(path)
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
    rows=[]
    for r in references():
        k=r['speaker'],r['condition'],r['scene'];candidates=index[k]
        item=dict(reference_id=r['filename'],speaker=k[0],condition=k[1],scene=k[2],candidates=[str(p.relative_to(ROOT)) for p in candidates])
        if len(candidates)!=1:
            item.update(status='ambiguous_source_recording' if candidates else 'source_recording_missing');rows.append(item);continue
        path=candidates[0]
        try:
            raw=path.read_bytes();text=raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')
            tiers,intervals=parse_chronological(text)
            if r['condition']=='L2' and len(tiers)==1:target=tiers[0]
            elif 'Speaker A' in tiers:target='Speaker A'
            else:raise ValueError('Target tier cannot be identified by documented conventions')
            labels=[i['label'] for i in intervals if i['tier']==target]
            clean,removed,payloads=cleaned_labels(labels)
            item.update(status='prepared',source_sha256=digest(path),target_tier=target,text=clean,
                text_sha256=hashlib.sha256(clean.encode()).hexdigest(),removed_annotation_counts=removed,
                retained_lexical_payload_counts=payloads,source_intervals=len(labels))
        except (ValueError,UnicodeError) as e:item.update(status='source_review_required',reason=str(e))
        rows.append(item)
    policy=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Direct paired comparison with actual per-sample published TAALED values from Baese-Berk et al. 2021, DOI 10.3389/fpsyg.2021.661415',
        reference_sha256=digest(DATA/'TAALED_results.csv'),source_files=sources,
        selection='All 240 published rows; exact speaker/condition/scene join from documented corpus names. Speaker A is first participant code, or the sole native-speaker tier for L2. S (street) corresponds to published shop scene.',
        ambiguity='Multiple candidate recordings remain missing; do not concatenate or choose by closest published scores. Every missing/ambiguous/review-required row retained.',
        preprocessing='One paragraph per target speaker. Explicit silence and pure angle-coded non-speech events removed. Angle-coded lexical payloads after underscore retained. Known embedded/malformed BR/LS/GA/LG/LA/SIL tags repaired, DEL/stray angle delimiters removed, all repairs counted; source files unchanged. Spelled fillers, repetitions and fragments follow existing unchanged production policy. Other unresolved annotations require review. Empty unused tier names are valid; target remains the named Speaker A or sole L2 tier.',
        implementations=['existing custom pipeline','taaled 0.32 / pylats 0.64 on identical prepared text'],
        metrics=list(SHARED),not_mapped=['root_ttr_aw (absent in published CSV)','mtld_mean_factor_length_aw (published moving-average MTLD variants are different measures)'],
        statistics='Per metric signed/absolute errors and APE. Recording means within speaker, then means/medians across speakers, per condition; no overall winner or pooled four-condition score when a condition is missing. Zero reference denominators unavailable. Exact counts, ratio match tolerance based on published decimal precision.',
        scope='Oracle source transcripts, no MOSS ASR or utterance segmentation ground truth. Current normalization/software may differ from original 2021 GUI; differences do not by themselves identify arithmetic defects.',
        frozen='No weight, normalization, cutoff, source-choice or routing changes from these results. No original ten-participant access.',
        preparation_counts=dict(Counter(r['status'] for r in rows)),unrecognized_sources=unrecognized)
    save(EVIDENCE/'protocol.json',policy);save(EVIDENCE/'prepared.json',rows)
    print(policy['preparation_counts'],flush=True)


def evaluate():
    import spacy
    from speech_pipeline.pipeline import lexical
    if (OUT/'results.json').exists():raise FileExistsError('Preserve completed published comparison')
    policy=json.loads((EVIDENCE/'protocol.json').read_text());prepared=json.loads((EVIDENCE/'prepared.json').read_text())
    if digest(DATA/'TAALED_results.csv')!=policy['reference_sha256']:raise ValueError('Published reference changed')
    for f in policy['source_files']:
        if digest(ROOT/f['path'])!=f['sha256']:raise ValueError('Source TextGrid changed')
    refs={r['filename']:r for r in references()};nlp=spacy.load('en_core_web_sm');pairs=[];coverage=[]
    with TemporaryDirectory(prefix='lucid_validation_') as temp:
        for i,r in enumerate(prepared):
            status=r['status'];measurements=None;reason=r.get('reason')
            if status=='prepared':
                try:
                    measurements=lexical([dict(text=r['text'])],nlp,Path(temp)/'input.txt');status='scored'
                except (ValueError,ZeroDivisionError) as e:status='pipeline_unscorable';reason=str(e)
            coverage.append(dict(reference_id=r['reference_id'],speaker=r['speaker'],condition=r['condition'],scene=r['scene'],status=status,reason=reason,candidates=r['candidates']))
            for metric in SHARED:
                string=refs[r['reference_id']][metric];truth=float(string) if string.strip() else None
                tolerance=0. if metric.startswith('basic_') else float(Decimal('0.5')*Decimal(10)**Decimal(string).as_tuple().exponent)+1e-12 if truth is not None else None
                for method,field in [('custom','custom'),('taaled_current','reference')]:
                    v=measurements[metric][field] if measurements else None
                    error=abs(v-truth) if v is not None and truth is not None else None
                    pairs.append(dict(reference_id=r['reference_id'],speaker=r['speaker'],condition=r['condition'],scene=r['scene'],method=method,metric=metric,
                        reference=truth,predicted=v,signed_error=v-truth if error is not None else None,absolute_error=error,
                        ape=100*error/abs(truth) if error is not None and truth else None,
                        within_published_precision=error<=tolerance if error is not None else None,status=status))
            if i%20==0:print('Published reference rows processed',i+1,'/',len(prepared),flush=True)
    grouped=defaultdict(lambda:defaultdict(list));matches=Counter();available=Counter();missing=Counter()
    for r in pairs:
        k=r['method'],r['condition'],r['metric']
        if r['ape'] is None:missing[k]+=1
        else:grouped[k][r['speaker']].append(r['ape']);available[k]+=1
        if r['within_published_precision']:matches[k]+=1
    summary=[]
    for method in ['custom','taaled_current']:
        for condition in ['noBarrier','vocoder','babble','L2']:
            for metric in SHARED:
                k=method,condition,metric;values=[float(np.mean(v)) for v in grouped[k].values()]
                summary.append(dict(method=method,condition=condition,metric=metric,
                    mean_speaker_mean_recording_ape=float(np.mean(values)) if values else None,
                    median_speaker_mean_recording_ape=float(np.median(values)) if values else None,
                    available=available[k],missing=missing[k],matching_published_precision=matches[k]))
    result=dict(protocol_sha256=digest(EVIDENCE/'protocol.json'),coverage_counts=dict(Counter(r['status'] for r in coverage)),
        coverage=coverage,summary=summary,pairs=pairs,versions={n:metadata.version(n) for n in ['spacy','en-core-web-sm','taaled','pylats','numpy']})
    save(OUT/'results.json',result)
    import csv
    for filename,rows in [('metric_pairs.csv',pairs),('metric_summary.csv',summary),('coverage.csv',coverage)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report(result)
    print('Coverage:',result['coverage_counts'],flush=True)


def report(result):
    lookup={(r['method'],r['condition'],r['metric']):r for r in result['summary']}
    lines=['# Direct published LUCID/TAALED comparison','',
        'Reference: Baese-Berk et al. (2021), DOI [10.3389/fpsyg.2021.661415](https://doi.org/10.3389/fpsyg.2021.661415), actual per-sample OSF TAALED output. '
        'Corpus TextGrids were obtained through the authorized SpeechBox data-use link and verified by ZIP-entry CRC and extracted SHA-256. '
        'Both current implementations receive the same prepared target-speaker words. No model or normalizer was tuned.','',
        f"All 240 published rows retained. Coverage: **{result['coverage_counts']}**. Ambiguous/missing source rows are unavailable, not zero-error or silently dropped.",'',
        '| Method | Condition | Scored rows | Token-count APE % | Type-count APE % | TTR APE % | MATTR50 APE % | HD-D APE % | Original MTLD APE % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in ['custom','taaled_current']:
        for condition in ['noBarrier','vocoder','babble','L2']:
            cells=[method,condition,str(lookup[method,condition,'basic_ntokens']['available'])]
            for k in ['basic_ntokens','basic_ntypes','simple_ttr_aw','mattr50_aw','hdd42_aw','mtld_original_aw']:
                v=lookup[method,condition,k]['mean_speaker_mean_recording_ape'];cells.append(f'{v:.2f}' if v is not None else 'unavailable')
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','APE is mean recording APE within speaker, then mean across speakers, for each condition. The full table retains all 15 shared metrics, signed errors, exact-precision match counts and missingness. '
        'The original MTLD and published moving-average MTLD variants are distinct; no unsupported field mapping was made.','',
        '## Interpretation','',
        'These are comparisons joined by documented speaker/condition/scene metadata; the exact historical cleaned inputs have not been recovered. '
        'They test oracle-transcript lexical measurement, not ASR, clinical diagnosis, human CHAT segmentation or whole-pipeline superiority. '
        'The original authors removed filled/unfilled pauses and vocal noises, but did not release the exact cleaned indexN.txt inputs or the full historical tool configuration here. '
        'Differences may therefore include source mapping, preprocessing, POS/lemmatization and software-version drift. They must not be described as exact reproduction or fixed by choosing favorable recordings. '
        'Vocoder has multiple source recordings per speaker/scene key and requires an author-supported mapping before a valid single-row comparison. '
        'No scored sample reproduces the published token count exactly with either current implementation. '
        'Large no-barrier count discrepancies also require source/tier mapping review, beyond assuming software drift. '
        'The table is a discrepancy audit, not evidence of exact reproduction or validated accuracy.','',
        'Source TextGrids are word intervals, not professional CHAT utterance gold. No MLU or segmentation score is inferred from their row count. '
        'Reference, source, prepared-input and software hashes and the frozen protocol are in `outputs/validation/lucid_validation/`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='evaluate':evaluate()
    else:report(json.loads((OUT/'results.json').read_text()))


if __name__=='__main__':main()
