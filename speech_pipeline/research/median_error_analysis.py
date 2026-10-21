"""Recompute publication medians from paired scores without modifying source labels.

Complete human samples use one APE per participant. Recording comparisons
use the median of participants' within-participant mean recording APEs.
This preserves the original within-participant aggregation. Bootstrap
participants jointly for both methods; intervals are percentile intervals.
"""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib, statistics, math
import numpy as np
from speech_pipeline import project_path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/results/human_comparison'
OUT=ROOT/'outputs/results/paper'
SEED=20261007
REPLICATES=10000

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    files=[SOURCE/'metric_pairs.csv',SOURCE/'participant_sample_pairs.csv',SOURCE/'boundary_recordings.csv',ROOT/'dataset/annotations/all_nafee_utterances.xlsx',OUT/'transcript_wer_participants.csv',OUT/'transcript_wer_recordings.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    audit=json.loads((ROOT/'outputs/validation/audits/error_summary/mean_median_analysis.json').read_text())
    changed = [name for name, h in audit['source_hashes'].items()
               if hashlib.sha256(project_path(name).read_bytes()).hexdigest() != h]
    if changed:
        raise ValueError('Saved audit source hashes do not match: ' + ', '.join(changed)
                         + '. Use the preserved audited scores for the current manuscript, or review and refresh the source audit after rescoring. See speech_pipeline/research/README.md.')
    pairs=read(files[0]);complete=read(files[1])
    for r in pairs+complete:
        x,y=float(r['pipeline_value']),float(r['reference_value'])
        if y:assert math.isclose(100*abs(x-y)/abs(y),float(r['absolute_percent_error']),abs_tol=1e-9)
    ids=sorted({r['patient_id'] for r in complete},key=lambda x:int(x[1:]));assert len(ids)==10
    samples=np.random.default_rng(SEED).integers(0,10,(REPLICATES,10))
    def interval(x):return list(map(float,np.quantile(x,[.025,.975])))
    def summary(rs):
        g=defaultdict(list)
        for r in rs:g[r['patient_id']].append(r)
        assert set(g)==set(ids)
        arrays={k:np.array([statistics.mean(float(r[f]) for r in g[p]) for p in ids]) for k,f in [('errors','absolute_percent_error'),('pipeline','pipeline_value'),('reference','reference_value'),('difference','difference')]}
        errors=arrays['errors'];boot=np.median(errors[samples],axis=1)
        return dict(mdape=float(np.median(errors)),mdape_ci=interval(boot),pipeline_median=float(np.median(arrays['pipeline'])),reference_median=float(np.median(arrays['reference'])),median_signed_difference=float(np.median(arrays['difference'])),median_signed_difference_ci=interval(np.median(arrays['difference'][samples],axis=1)),minimum=float(errors.min()),maximum=float(errors.max()),participant_errors=dict(zip(ids,map(float,errors))),n_participants=len(g),n_score_pairs=len(rs)),boot
    data=json.loads((ROOT/'outputs/results/paper/publication_data.json').read_text())
    result=dict(bootstrap=dict(seed=SEED,replicates=REPLICATES,confidence_level=.95,method='Participant-cluster percentile bootstrap of median participant errors'),definition='Complete samples: median of participant-sample APEs. Recording comparisons: median of within-participant mean recording APEs.',complete_samples={},recording_samples={},corrected={},baseline={},sources=hashes)
    for row in data['tables']:
        m=row['metric'];sm=row['software_metric'];ref=row['software_reference']
        for key,rs,comp,bucket in [
            ('manual',[r for r in complete if r['metric']==m and r['method']=='Revised'],'Nafee complete participant samples','complete_samples'),
            ('manual_recording_segments',[r for r in pairs if r['metric']==m and r['method']=='Revised' and r['reference']=='Nafee'],'Nafee recording comparison','recording_samples'),
            ('software',[r for r in pairs if r['metric']==sm and r['method']=='Revised' and r['reference']==ref],ref+' recording comparison','corrected')]:
            s,_=summary(rs);row[key].update(s);result[bucket][sm if key=='software' else m]=s
            a=next(r for r in audit['metrics'] if r['method']=='Revised' and r['comparison']==comp and r['metric']==(sm if key=='software' else m))
            assert math.isclose(s['mdape'],a['median_participant_error_percent'],abs_tol=1e-10)
        if m in ['total_utterances','mlu_words','mlu_words_sd']:
            old,ob=summary([r for r in complete if r['metric']==m and r['method']=='Previous'])
            cur,cb=summary([r for r in complete if r['metric']==m and r['method']=='Revised'])
            result['baseline'][m]=dict(previous=old,current=cur,reduction=old['mdape']-cur['mdape'],reduction_ci=interval(ob-cb))
    wers=read(OUT/'transcript_wer_participants.csv');v={r['patient_id']:float(r['mean_recording_wer_percent']) for r in wers};a=np.array([v[p] for p in ids])
    result['wer']=dict(median_participant_mean=float(np.median(a)),ci=interval(np.median(a[samples],axis=1)))
    for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))]
    data['error_summary']='Median participant error; original mean summaries retained for audit only'
    (ROOT/'outputs/results/paper/publication_data.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'median_robustness_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for r in data['tables']:
        for name,key in [('Corrected software','software'),('Nafee complete participant sample','manual'),('Nafee recording segments','manual_recording_segments')]:
            s=r[key];rows.append(dict(metric=r['software_metric'] if key=='software' else r['metric'],label=r['label'],comparison=name,pipeline_median=s['pipeline_median'],reference_median=s['reference_median'],median_participant_error_percent=s['mdape'],ci_lower=s['mdape_ci'][0],ci_upper=s['mdape_ci'][1]))
    with (OUT/'publication_tables_median.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('WER',result['wer'])
    for m in ['total_utterances','mlu_words','mlu_words_sd']:print(m,json.dumps(result['baseline'][m]))
    for m in ['simple_ttr_aw','mattr50_aw','mlu_words','mlu_words_sd']:print('corrected',m,result['corrected'][m])

if __name__=='__main__':main()
