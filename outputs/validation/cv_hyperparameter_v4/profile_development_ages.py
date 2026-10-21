"""Descriptive CHAT target-child ages, original train/eval only; no selection."""
import json,re
from pathlib import Path
import numpy as np
from speech_pipeline import ROOT
from speech_pipeline.cohorts import load_split

def run():
    children={}
    for row in load_split('train')+load_split('validation'):
        if row['participant'] in children:continue
        path=ROOT/'dataset/public'/row['corpus']/row['recording'].split('#section')[0]
        ages=[]
        for line in path.read_text(errors='replace').splitlines():
            if not line.startswith('@ID:'):continue
            fields=line.split(':',1)[1].strip().split('|')
            if len(fields)<4 or fields[2]!='CHI':continue
            match=re.fullmatch(r'(\d+);(\d+)(?:\.(\d*))?',fields[3])
            if match:ages.append(int(match[1])+int(match[2])/12+int(match[3] or 0)/365.25)
        children[row['participant']]=dict(corpus=row['corpus'],age_years=ages[0] if len(ages)==1 else None)
    summary=[]
    for corpus in ['nyu_emerson','ocsc']:
        values=[v['age_years'] for v in children.values() if v['corpus']==corpus and v['age_years'] is not None]
        summary.append(dict(corpus=corpus,participants=sum(v['corpus']==corpus for v in children.values()),age_available=len(values),
            median_years=float(np.median(values)),min_years=min(values),max_years=max(values)))
    result=dict(scope='Descriptive original train+eval CHAT @ID target-child ages only. No test metadata, model features, hyperparameter/grid or cohort changes; differences do not prove causation.',summary=summary)
    path=Path(__file__).with_name('development_age_profile.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
