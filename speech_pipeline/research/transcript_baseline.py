"""Isolate scoring-policy changes using unchanged saved test predictions."""
import json
from concurrent.futures import ThreadPoolExecutor
from speech_pipeline import ROOT
from speech_pipeline.pipeline import save,digest
from speech_pipeline.chat_metrics import analyze,native_values
from speech_pipeline.research.transcript_chat import OUT,EVIDENCE,summarize,METRICS


def main():
    path=OUT/'chat_same_predictions.json'
    if path.exists():raise FileExistsError('Preserve baseline decomposition')
    old_path=ROOT/'outputs/results/common_benchmark/original_chat_sensitivity.json'
    old=json.loads(old_path.read_text())
    refs={(r['corpus'],r['recording']):native_values(r['official_outputs']['mlu'],r['official_outputs']['freq']) for r in old['references']}
    pairs={(r['corpus'],r['recording'],r['metric']):r['predicted'] for r in old['pairs'] if r['method']=='latest_minilm'}
    frozen=json.loads((ROOT/'outputs/results/common_benchmark/results.json').read_text())['recordings']
    selected=[r for r in frozen if r['method']=='latest_minilm'];rows=[]
    def score(r):
        us=[dict(text=t) for t in r['utterances']]
        native=analyze(us)['values'];ref=dict(refs[r['corpus'],r['recording']],total_utterances=None)
        keys=dict(total_utterances=None,eligible_mlu_utterances='eligible_mlu_utterances',MLUw='MLUw',length_sd='length_sd',mlu_words='mlu_words',TNW='freq/TNW',NDW='freq/NDW',TTR='freq/TTR',MATTR='freq/MATTR')
        before={k:pairs.get((r['corpus'],r['recording'],v)) if v else None for k,v in keys.items()}
        identity={k:r[k] for k in ['corpus','participant','recording']}
        return [dict(**identity,method='previous_scoring_same_predictions',reference=ref,predicted=before),dict(**identity,method='native_scoring_same_predictions',reference=ref,predicted=native)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for x in pool.map(score,selected):rows.extend(x)
    save(path,dict(scope='Unchanged test utterances and source words; only metric policy changes. This is a descriptive reference-processing audit, not model selection.',source_sha256=digest(old_path),records=rows,summary=summarize(rows)))
    print('Decomposed unchanged predictions:',len(selected),flush=True)

if __name__=='__main__':main()
