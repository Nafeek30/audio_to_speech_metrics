"""Source-supported historical profile comparison; retain previous results."""
import argparse,ast,csv,hashlib,json,math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
from speech_pipeline import ROOT
from speech_pipeline.pipeline import save,digest
from speech_pipeline.lexical_compat import HistoricalTaaled,clean_pauses,calculate_normalized
from speech_pipeline.research.lucid_validation import SHARED,references

OUT=ROOT/'outputs/results/transcript_validation'
EVIDENCE=ROOT/'outputs/validation/transcript_validation'
LEGACY=ROOT/'outputs/validation/lucid_validation/upstream_gui'
PREPARED=ROOT/'outputs/validation/lucid_validation/prepared.json'


def register():
    p=EVIDENCE/'protocol.json'
    if p.exists():raise FileExistsError('Preserve registered comparison')
    source_files=[LEGACY/n for n in ['TAALED_1_4_1.py','real_words.txt','adj_lem_list.txt']]
    inventory=[]
    for corpus in ['ocsc','nyu_emerson']:
        files=sorted((ROOT/'dataset/public'/corpus/'transcripts').rglob('*.cha'))
        inventory.append(dict(corpus=corpus,transcripts=len(files),files=[dict(path=str(f.relative_to(ROOT)),sha256=digest(f)) for f in files]))
    plan=dict(registered_utc=datetime.now(timezone.utc).isoformat(),scope='Transcript-only LUCID/OCSC/NYU; audio and original ten participants deferred.',
        purpose='Correct a documented profile mismatch: official historical dictionary and tag-dependent lexical identities instead of assuming current pylats is historical TAALED.',
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in source_files],
        input_sha256=digest(PREPARED),published_sha256=digest(ROOT/'dataset/public/lucid_published/TAALED_results.csv'),
        lexical_policy='Historical 1.4.1 dictionary, POS/category/noun-verb/pronoun/adverb rules ported without score fitting; fixed spelled-pause list removed consistent with paper pause-removal policy. Same metadata/source mapping as previous run; no selecting among ambiguous vocoder recordings.',
        limitations='Current spaCy 3.8 model, not original historical model; exact original cleaned files/configuration unknown. Err treated as pause; lexical uses of err would need review. Descriptive profile correction on already exposed references, not independent confirmation.',
        calculator_check='Compare our independent formulas and ported token rules against AST-extracted official GUI pure functions on identical docs/tokens. Historical GUI application is never launched. This is implementation parity, distinct from published agreement.',
        aggregation='Mean recording APE within speaker, then equally across speakers, separately per condition/corpus; no zero filling or omitted failure rows.',
        engineering_targets='Declared before corrected scoring: mean participant APE <=5% for counts/densities/TTR/MATTR/MSTTR/HD-D and MLUw; <=10% for length SD/original MTLD. These are project aims, not clinical acceptance standards. All metrics and failures reported; no pooled score to hide failures.',
        frozen='No reference-driven source remapping, forced counts, model/threshold selection or final-data training. Previous results and weights preserved.',
        inventories=inventory,code={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'speech_pipeline/lexical_compat.py',Path(__file__).resolve()]})
    save(p,plan);print('Registered; transcripts:',[(x['corpus'],x['transcripts']) for x in inventory],flush=True)


def upstream_functions():
    """Execute inspected pure functions only, never imports/GUI/file operations."""
    source=LEGACY/'TAALED_1_4_1.py';tree=ast.parse(source.read_text())
    names={'safe_divide','ttr','mattr','msttr','hdd','mtld_original','tag_processor_spaCy'}
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('Upstream function set changed')
    env=dict(math=math,Counter=Counter,adj_word_list=set((LEGACY/'adj_lem_list.txt').read_text().splitlines()),real_word_list=set((LEGACY/'real_words.txt').read_text().splitlines()))
    constants={'noun_tags','content','prelim_not_function','pronouns','pronoun_dict','verbs','adverbs','punctuation'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in constants:
            # punctuation is a literal split expression in official source.
            if node.targets[0].id=='punctuation':env['punctuation']="`` '' ' . , ? ! ) ( % / - _ -LRB- -RRB- SYM : ;".split()+['"']
            else:env[node.targets[0].id]=ast.literal_eval(node.value)
    code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(code,str(source),'exec'),env)
    return env


def evaluate():
    import spacy
    if (OUT/'lucid_results.json').exists():raise FileExistsError('Preserve completed comparison')
    plan=json.loads((EVIDENCE/'protocol.json').read_text())
    assert digest(PREPARED)==plan['input_sha256']
    for x in plan['sources']:assert digest(ROOT/x['path'])==x['sha256']
    nlp=spacy.load('en_core_web_sm');profile=HistoricalTaaled(LEGACY);official=upstream_functions()
    rows=json.loads(PREPARED.read_text());refs={r['filename']:r for r in references()}
    ready=[r for r in rows if r['status']=='prepared'];cleaned=[clean_pauses(r['text']) for r in ready]
    predictions={};checks=[]
    for i,(r,doc,(text,removed)) in enumerate(zip(ready,nlp.pipe((x[0] for x in cleaned),batch_size=8),cleaned)):
        normal=profile.normalize(doc);mine=calculate_normalized(normal)
        official['nlp']=lambda _,doc=doc:doc
        theirs=official['tag_processor_spaCy'](text)
        if any(normal[k]!=theirs[k] for k in ('lemma','content','function')):raise ValueError('Historical token-rule port mismatch')
        toks=theirs['lemma'];ttr=official['ttr'](toks)
        expected=dict(simple_ttr_aw=ttr[0],log_ttr_aw=ttr[2],maas_ttr_aw=ttr[3],
            mattr50_aw=official['mattr'](toks),msttr50_aw=official['msttr'](toks),
            hdd42_aw=official['hdd'](toks),mtld_original_aw=official['mtld_original'](toks))
        for metric,v in expected.items():
            if mine[metric] is not None and not math.isclose(mine[metric],v,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError(f'Historical calculator mismatch: {r["reference_id"]}, {metric}, {mine[metric]}, {v}')
        checks.append(dict(reference_id=r['reference_id'],token_rules_match=True,formula_checks=len(expected),passed=True))
        predictions[r['reference_id']]=dict(values=mine,normalized_tokens=normal['lemma'],pause_removals=removed,excluded_token_counts=normal['excluded_token_counts'])
        if i%30==0:print('Historical profile processed',i+1,'/',len(ready),flush=True)
    pairs=[];coverage=[]
    for r in rows:
        pred=predictions.get(r['reference_id']);status='scored' if pred else r['status']
        coverage.append(dict(reference_id=r['reference_id'],speaker=r['speaker'],condition=r['condition'],status=status,candidates=r['candidates']))
        for metric in SHARED:
            truth=float(refs[r['reference_id']][metric]);value=pred['values'][metric] if pred else None
            pairs.append(dict(reference_id=r['reference_id'],speaker=r['speaker'],condition=r['condition'],metric=metric,
                reference=truth,predicted=value,absolute_error=abs(value-truth) if value is not None else None,
                signed_error=value-truth if value is not None else None,ape=100*abs(value-truth)/abs(truth) if value is not None and truth else None,status=status))
    grouped=defaultdict(lambda:defaultdict(list))
    for r in pairs:
        if r['ape'] is not None:grouped[r['condition'],r['metric']][r['speaker']].append(r['ape'])
    summary=[]
    for condition in ['noBarrier','babble','L2','vocoder']:
        for metric in SHARED:
            vals=grouped[condition,metric];speakermeans=[float(np.mean(v)) for v in vals.values()]
            target=10 if metric=='mtld_original_aw' else 5
            error=float(np.mean(speakermeans)) if speakermeans else None
            summary.append(dict(condition=condition,metric=metric,available=sum(map(len,vals.values())),speakers=len(vals),mean_participant_ape=error,target_ape=target,meets_engineering_target=error<=target if error is not None else None))
    result=dict(protocol_sha256=digest(EVIDENCE/'protocol.json'),coverage_counts=dict(Counter(r['status'] for r in coverage)),coverage=coverage,
        summary=summary,pairs=pairs,predictions=predictions,calculator_checks=checks,versions={n:version(n) for n in ['spacy','en-core-web-sm','numpy']},
        scope=plan['limitations'])
    save(OUT/'lucid_results.json',result)
    for name,data in [('lucid_metric_pairs.csv',pairs),('lucid_summary.csv',summary),('lucid_coverage.csv',coverage)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print('Coverage',result['coverage_counts'],flush=True)
    for r in summary:
        if r['metric'] in {'basic_ntokens','simple_ttr_aw','mattr50_aw','hdd42_aw','mtld_original_aw'}:print(r,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','evaluate']);a=p.parse_args()
    register() if a.stage=='register' else evaluate()

if __name__=='__main__':main()
