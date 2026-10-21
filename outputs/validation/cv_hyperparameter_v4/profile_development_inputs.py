"""Descriptive source-profile comparison using original train/eval only."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from speech_pipeline.cohorts import load_split
from speech_pipeline.research import participant_cv as cv
from speech_pipeline.pipeline import save

training=load_split('train');rows=training+load_split('validation');grouped=defaultdict(list)
training_summary=[dict(corpus=c,participants=len({r['participant'] for r in training if r['corpus']==c}),
 sections=sum(r['corpus']==c for r in training),source_words=sum(len(r['tokens']) for r in training if r['corpus']==c),
 possible_nonfinal_gaps=sum(len(r['tokens'])-1 for r in training if r['corpus']==c),
 supervised_gaps=sum(sum(v>=0 for v in r['labels']) for r in training if r['corpus']==c),
 supervised_positive_boundaries=sum(sum(v==1 for v in r['labels']) for r in training if r['corpus']==c))
 for c in sorted({r['corpus'] for r in training})]
for r in training_summary:r['masked_nonfinal_gap_percent']=100*(1-r['supervised_gaps']/r['possible_nonfinal_gaps'])
for r in rows:
 path=cv.EVIDENCE/'prepared'/(hashlib.sha256(cv.key(r).encode()).hexdigest()+'.json')
 prepared=json.loads(path.read_text());contexts=prepared['contexts'];ref=prepared['reference'];lengths=[len(c['tokens']) for c in contexts]
 grouped[r['corpus']].append(dict(participant=r['participant'],words=len(r['tokens']),contexts=len(contexts),short_contexts=sum(n<=3 for n in lengths),
  context_lengths=lengths,reference_verbal_units=ref['total_utterances'],reference_eligible_units=ref['eligible_mlu_utterances'],
  reference_mlu_words=ref['mlu_words'],reference_freq_words=ref['TNW'],
  unintelligibility_words=sum(t.casefold() in {'xxx','yyy','www'} for t in r['tokens']),
  pause_words=sum(t.casefold() in {'uh','um'} for t in r['tokens'])))
summary=[]
for corpus,rr in grouped.items():
 lengths=[n for r in rr for n in r['context_lengths']];tokens=sum(r['words'] for r in rr);mlu=sum(r['reference_mlu_words'] for r in rr);freq=sum(r['reference_freq_words'] for r in rr)
 summary.append(dict(corpus=corpus,participants=len({r['participant'] for r in rr}),sections=len(rr),source_words=tokens,
  median_words_per_section=float(np.median([r['words'] for r in rr])),median_words_per_observed_turn=float(np.median(lengths)),
  percent_observed_turns_at_most_three_words=100*sum(r['short_contexts'] for r in rr)/len(lengths),
  unintelligibility_tokens_per_thousand_source_words=1000*sum(r['unintelligibility_words'] for r in rr)/tokens,
  uh_um_tokens_per_thousand_source_words=1000*sum(r['pause_words'] for r in rr)/tokens,
  reference_freq_words=freq,reference_mlu_words=mlu,reference_percent_freq_words_absent_from_mlu=100*(freq-mlu)/freq,
  sections_with_more_observed_turns_than_human_verbal_units=sum(r['contexts']>r['reference_verbal_units'] for r in rr)))
save(cv.ROOT/'outputs/validation/cv_hyperparameter_v4/development_input_profile.json',dict(scope='Original train+eval only. Descriptive development profile; source data and cohorts remain unchanged. Current CV grid registered separately.',training_summary=training_summary,summary=summary,
 interpretation='Observed turns are input contexts, not gold utterances. Short turns, unintelligibility and native FREQ/MLU eligibility differences are descriptive indicators; they do not prove a causal explanation for model errors. Verbal-unit floor counts are distinct from native eligible MLU counts.'))
print(json.dumps(summary,indent=2))
