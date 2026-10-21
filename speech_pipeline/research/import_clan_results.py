"""Import returned CLAN outputs and rescore corrected text with TAALED."""
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[2]
import spacy
import speech_pipeline.utterances as auto
from speech_pipeline.pipeline import lexical,clan_reference,digest,save,SOURCE_FILES
from speech_pipeline.export import export_all

base=ROOT/'outputs/validation/clan/same_boundaries'
out=ROOT/'outputs/validation/method_comparison/analysis/corrected'
previous=json.loads((out/'comparison.json').read_text())
prior={(r['patient_id'],r['source_file']):r for r in previous['recordings']}
rows=list(csv.DictReader((base/'manifest.csv').open(encoding='utf-8-sig')))
nlp=spacy.load('en_core_web_sm')
report=dict(run_utc=datetime.now(timezone.utc).isoformat(),dataset_type='corrected',method=auto.METHOD,
    scope='Corrected transcripts with the same clause-based splitter as the uncorrected pipeline; source turns mapped by word alignment',
    scoring_policy=auto.POLICY,versions={n:version(n) for n in ('spacy','en-core-web-sm','taaled','pylats')},
    code_sha256={p:digest(ROOT/p) for p in SOURCE_FILES},
    corrected_export_provenance=json.loads((base/'provenance.json').read_text()),recordings=[],patients=[])
pooled=defaultdict(list)
with TemporaryDirectory() as tmp:
    for row in rows:
        data=json.loads((base/row['utterance_json']).read_text())
        key=(row['patient_id'],row['corrected_source'])
        old=prior[key]
        source=ROOT/'dataset/corrected'/key[0]/key[1]
        chat=base/'corrected_auto'/row['chat_filename']
        result=chat.with_suffix('.mlu.cex')
        assert digest(source)==data['corrected_sha256']
        assert digest(chat)==row['corrected_chat_sha256']
        utterances=data['utterances'];mlu=auto.word_mlu(utterances)
        assert mlu==data['corrected_metrics']
        clan=clan_reference(result,chat.stem,mlu)
        assert all(v['absolute_error']==0 for k,v in clan['metrics'].items() if k in ['mlu_nutterances','mlu_nwords'])
        assert all(v['absolute_error']<=.000501 for k,v in clan['metrics'].items() if k in ['mlu_words','mlu_words_sd'])
        sha=digest(chat)
        chat_file=f'clan_software_input_files/{key[0]}_{Path(key[1]).stem}_{row["target_speaker"]}_auto_{sha[:12]}.cha'
        target=out/chat_file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(chat.read_bytes())
        save(out/old['utterance_file'],dict(method=auto.METHOD,raw_sha256=digest(source),
            source_file=f'{key[0]}/{key[1]}',target_speaker=row['target_speaker'],
            turn_mapping=data['turn_alignment'],utterances=utterances))
        clan.update(input_file=str(chat.relative_to(ROOT)),input_sha256=digest(chat),
                    output_file=str(result.relative_to(ROOT)),software_version='31-Aug-2026')
        rec=dict(patient_id=key[0],source_file=key[1],target_speaker=row['target_speaker'],
            raw_sha256=digest(source),source_rows=old['source_rows'],
            utterance_file=old['utterance_file'],prediction_file=old['utterance_file'],
            chat_file=chat_file,chat_sha256=sha,mlu=mlu,
            lexical=lexical(utterances,nlp,Path(tmp)/'sample.txt'),clan=clan,
            turn_mapping_review_required=sum(a['review_required'] for a in data['turn_alignment']))
        report['recordings'].append(rec);pooled[key[0]].extend(utterances)
        print(f'Imported CLAN and reran TAALED: {key[0]}/{key[1]}',flush=True)
    for pid,utterances in pooled.items():
        report['patients'].append(dict(patient_id=pid,pool_order='source filename ascending',
            mlu=auto.word_mlu(utterances),lexical=lexical(utterances,nlp,Path(tmp)/'pool.txt')))
save(out/'comparison.json',report)
current={r['chat_file'] for r in report['recordings']}
for p in (out/'clan_software_input_files').glob('*_auto_*.cha'):
    if str(p.relative_to(out)) not in current:
        p.unlink()
export_all(report,out)
print('Imported all 37 returned CLAN outputs and rescored corrected text with TAALED.')
