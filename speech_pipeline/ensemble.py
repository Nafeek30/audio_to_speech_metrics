"""Equal-probability ensembles of independently trained gap models."""
import json
from pathlib import Path

import numpy as np

from speech_pipeline import ROOT, utterances as auto
from speech_pipeline.pipeline import digest
from speech_pipeline.segmentation import Segmenter, pretrained_logits


def mean_probability_logits(values):
    """Return boundary log odds after equal-weight probability averaging.

    A singleton preserves its exact logits. Never average independent task-head
    weights, and never weight ensemble members using held-out test scores.
    """
    arrays=[np.asarray(v,dtype=np.float64) for v in values]
    if not arrays:raise ValueError('An ensemble needs at least one member')
    if any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Ensemble emissions must have identical finite shapes')
    if len(arrays)==1:return arrays[0].copy()
    probabilities=np.mean([np.exp(-np.logaddexp(0.,-v)) for v in arrays],axis=0)
    probabilities=np.clip(probabilities,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return np.log(probabilities)-np.log1p(-probabilities)


class EnsembleSegmenter(Segmenter):
    def __init__(self,folder,nlp=None,device='auto'):
        self.folder=Path(folder);self.metadata=json.loads((self.folder/'ensemble.json').read_text())
        if self.metadata['format']!='gap_probability_ensemble_v1':raise ValueError('Unsupported ensemble')
        self.members=[]
        for item in self.metadata['members']:
            path=(ROOT/item['model_directory']).resolve()
            if not path.is_relative_to(ROOT.resolve()):raise ValueError('Ensemble member is outside project')
            if digest(path/'model.safetensors')!=item['weights_sha256']:raise ValueError('Ensemble member weights changed')
            member=Segmenter(path,nlp,device);self.members.append(member);nlp=member.nlp
        if not self.members:raise ValueError('Empty ensemble')
        self.nlp=nlp;self.device=self.members[0].device

    def probabilities(self,tokens,rules=None):
        if not tokens:return np.array([])
        rules=self.features(tokens) if rules is None else rules
        records=[dict(tokens=tokens,baseline=rules)]
        v=self._logits(records)[0]
        return np.exp(-np.logaddexp(0.,-v))

    def _logits(self,records):
        emissions=[]
        for member in self.members:
            values=pretrained_logits(member.model,member.tokenizer,records,
                member.metadata.get('max_length',256),member.metadata.get('stride',128),member.metadata.get('inference_batch_size',8))
            emissions.append([v[:,1]-v[:,0] for v in values])
        return [mean_probability_logits(v) for v in zip(*emissions)]

    def segment(self,text,boundary_margin=None):
        return self.segment_many([text],boundary_margin)[0]

    def segment_many(self,texts,boundary_margin=None):
        margin=self.metadata['margin'] if boundary_margin is None else boundary_margin
        if not np.isfinite(margin):raise ValueError('A finite boundary margin is required')
        texts=list(texts);matches=[list(auto.WORD.finditer(t)) for t in texts]
        active=[i for i,m in enumerate(matches) if m];records=[]
        for i in active:
            tokens=[m.group().casefold().replace('’',"'") for m in matches[i]]
            records.append(dict(tokens=tokens,baseline=self.features(tokens)))
        output=[[] for _ in texts]
        if not active:return output
        for i,values in zip(active,self._logits(records)):
            cuts=[0,*[matches[i][j+1].start() for j,v in enumerate(values[:-1]) if v>margin],len(texts[i])]
            for a,b in zip(cuts,cuts[1:]):
                while a<b and texts[i][a].isspace():a+=1
                while b>a and texts[i][b-1].isspace():b-=1
                body=texts[i][a:b]
                output[i].append(dict(utterance_id=len(output[i])+1,text=body,source_start=a,source_end=b,
                    reviewed=False,method='gap_probability_ensemble',scoring=auto.score_utterance(body)))
        return output
