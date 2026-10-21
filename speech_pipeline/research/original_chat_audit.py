"""Sensitivity audit against native CLAN on original professional CHAT coding.

This is a separate, explicitly unharmonized reference-policy analysis. It does
not replace the frozen normalized benchmark or choose any model. Native FREQ
and MLU have different eligibility policies; retain that difference in labels.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from speech_pipeline import ROOT
from speech_pipeline.corpora import read_chat, run_clan
from speech_pipeline.pipeline import digest, save
from speech_pipeline.research.common_benchmark import OUT, EVIDENCE, TEST, key


def section_chat(source, section):
    """Preserve original main-tier coding; no MOR is supplied to word commands."""
    sections=[]
    for row in read_chat(source):
        if not sections or row['gem']!=sections[-1][0]['gem']:
            sections.append([])
        sections[-1].append(row)
    rows=sections[section-1]
    header=[]
    for line in source.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('*'):break
        if line.startswith(('@UTF8','@Begin','@Languages:','@Participants:','@ID:')):
            header.append(line)
    return '\n'.join(header+[f"*{r['speaker']}:\t{r['raw_chat']}" for r in rows]+['@End','']),rows[0]['gem']


def main():
    path=OUT/'original_chat_sensitivity.json'
    if path.exists():raise FileExistsError('Preserve completed reference-policy sensitivity audit')
    protocol=EVIDENCE/'original_chat_protocol.json'
    if not protocol.exists():
        save(protocol,dict(registered_utc=datetime.now(timezone.utc).isoformat(),
            primary_protocol='protocol.json remains unchanged; no training, choice or tuning',
            purpose='Quantify reference-policy sensitivity using original annotated CHAT rather than only normalized gold words',
            scope='All 126 public test task sections; native CLAN MLU -t%mor and FREQ +k +b50 on original main tiers, target CHI',
            measures='Eligible MLU utterances (distinct from total units), word MLU, population length SD, MLU eligible words, FREQ TNW/NDW/TTR/MATTR50',
            comparisons='Frozen system predictions and frozen normalized gold; all records retained. Native FREQ word policy differs from clinical TNW policy; agreement reflects both coding and segmentation differences.',
            endpoints='Mean recording APE within child, mean child within corpus, no new composite or selection; zero denominators missing',
            input_sha256=digest(TEST),predictions_sha256=digest(OUT/'results.json')))
    plan=json.loads(protocol.read_text())
    if plan['input_sha256']!=digest(TEST) or plan['predictions_sha256']!=digest(OUT/'results.json'):
        raise ValueError('Frozen inputs changed')
    records=[json.loads(s) for s in TEST.read_text().splitlines()]
    result=json.loads((OUT/'results.json').read_text())
    predictions=defaultdict(list)
    for row in result['recordings']:predictions[key(row)].append(row)
    refs=[];pairs=[]
    from speech_pipeline.research.common_benchmark import units
    from speech_pipeline import utterances as auto
    from speech_pipeline.metrics import clinical_from_predictions
    def metrics(us):
        a=auto.word_mlu(us);b=clinical_from_predictions(us)['values']
        return dict(eligible_mlu_utterances=a['mlu_nutterances'],MLUw=a['mlu_words'],
                    length_sd=a['mlu_words_sd'],mlu_words=a['mlu_nwords'],
                    **{f'freq/{k}':b[k] for k in ('TNW','NDW','TTR','MATTR')})
    with TemporaryDirectory(prefix='original_chat_reference_') as temp:
        temp=Path(temp)
        for i,r in enumerate(records):
            filename,section=r['recording'].rsplit('#section',1)
            source=ROOT/'dataset/public'/r['corpus']/filename
            if digest(source)!=r['source_sha256']:raise ValueError('Original source changed')
            text,gem=section_chat(source,int(section))
            if gem!=r['gem']:raise ValueError('Reference task alignment changed')
            chat=temp/'original.cha';chat.write_text(text)
            mlu=run_clan('mlu',chat,speaker='CHI')
            freq=run_clan('freq',chat,speaker='CHI',window=50)
            reference=dict(eligible_mlu_utterances=mlu['utterances'],MLUw=mlu['mlu'],
                length_sd=mlu['sd'],mlu_words=mlu['total'],**{f'freq/{k}':freq[k] for k in ('TNW','NDW','TTR','MATTR')})
            # CLAN may print a MATTR number for a sequence shorter than its
            # declared window; this benchmark requires a full 50-word window.
            if reference['freq/TNW'] is None or reference['freq/TNW']<50:reference['freq/MATTR']=None
            refs.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],
                original_section_sha256=digest(chat),original_chat=text,reference=reference,official_outputs=dict(mlu=mlu,freq=freq)))
            conditions=[('normalized_gold',metrics(units(r['surface_words'],r['gold_boundaries'])))]
            conditions += [(p['method'],metrics([dict(text=t) for t in p['utterances']])) for p in predictions[key(r)]]
            for method,mine in conditions:
                for k,v in reference.items():
                    p=mine[k]
                    pairs.append(dict(corpus=r['corpus'],participant=r['participant'],recording=r['recording'],method=method,metric=k,
                        reference=v,predicted=p,signed_error=p-v if p is not None and v is not None else None,
                        ape=100*abs(p-v)/abs(v) if p is not None and v else None))
            if i%25==0:print(f'Native original CHAT references {i+1}/{len(records)}',flush=True)
    groups=defaultdict(lambda:defaultdict(list));available=defaultdict(int);missing=defaultdict(int)
    for r in pairs:
        k=(r['method'],r['corpus'],r['metric'])
        if r['ape'] is None:missing[k]+=1
        else:groups[k][r['participant']].append(r['ape']);available[k]+=1
    summary=[dict(method=k[0],corpus=k[1],metric=k[2],mean_child_mean_recording_ape=float(np.mean([np.mean(v) for v in children.values()])),
                  available_recordings=available[k],missing_recordings=missing[k]) for k,children in sorted(groups.items())]
    save(path,dict(protocol_sha256=digest(protocol),references=refs,pairs=pairs,summary=summary,
                   caveat='Original coding and normalized word policy are different. This sensitivity audit does not establish that the clinical surface profile equals full coded CHAT/FREQ.'))
    import csv
    for filename,rows in [('original_chat_metric_pairs.csv',pairs),('original_chat_summary.csv',summary)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Original coded CHAT reference sensitivity','',
        'Native CLAN scores original professional main-tier codes for exactly the same public task sections. '
        'This preserves retraces, fillers, replacements, exclusions and continuations. Word MLU uses `-t%mor`. '
        'FREQ and clinical surface-word eligibility differ, so those differences are reported as policy sensitivity. '
        'No model or normalized-benchmark endpoint was changed.','',
        '| Condition | Corpus | Eligible-count APE % | MLUw APE % | SD APE % | FREQ TNW agreement APE % | FREQ NDW agreement APE % |',
        '|---|---|---:|---:|---:|---:|---:|']
    lookup={(r['method'],r['corpus'],r['metric']):r for r in summary}
    for m in ['normalized_gold','rules','previous_minilm','latest_minilm','batchalign_0.10.4']:
        for c in ['ocsc','nyu_emerson']:
            numbers=[lookup[m,c,k]['mean_child_mean_recording_ape'] for k in ['eligible_mlu_utterances','MLUw','length_sd','freq/TNW','freq/NDW']]
            lines.append('| '+' | '.join([m,c]+[f'{n:.2f}' for n in numbers])+' |')
    lines+=['','`normalized_gold` isolates the change caused by removing CHAT coding and applying our surface profile, while retaining human boundaries. '
        'Its nonzero error is a reference-processing difference before segmentation errors. Native MLU eligible utterance count is not total utterance count. '
        'CLAN reference ratios/SD are rounded to three decimals. All reference files, commands and outputs are in `original_chat_sensitivity.json`; '
        'every metric pair and missing result is retained in CSV.','']
    (OUT/'ORIGINAL_CHAT_AUDIT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
