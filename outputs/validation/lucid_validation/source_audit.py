"""Post-score provenance checks; never alter source matching or predictions."""
import csv,hashlib,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean
from speech_pipeline import ROOT
base=ROOT/'dataset/public/lucid_published'
evidence=ROOT/'outputs/validation/lucid_validation'
rows=list(csv.DictReader((base/'TAALED_results.csv').open(encoding='utf-8-sig')))
results=json.loads((ROOT/'outputs/results/lucid_validation/results.json').read_text())
prepared=json.loads((evidence/'prepared.json').read_text())
article=dict(noBarrier=662.78,L2=1095.92,babble=756.75,vocoder=785.21)
means=[]
for condition,published in article.items():
 observed=mean(float(r['basic_ntokens']) for r in rows if r['condition']==condition)
 means.append(dict(condition=condition,article_table1_tokens=published,supplement_mean_tokens=observed,absolute_difference=abs(observed-published),within_0_01_word=abs(observed-published)<.01))
ratio_checks=[]
for r in rows:
 truth=float(r['simple_ttr_aw']);derived=int(r['basic_ntypes'])/int(r['basic_ntokens'])
 tolerance=float(Decimal('0.5')*Decimal(10)**Decimal(r['simple_ttr_aw']).as_tuple().exponent)+1e-12
 ratio_checks.append(dict(reference_id=r['filename'],absolute_error=abs(derived-truth),within_published_precision=abs(derived-truth)<=tolerance))
lookup={r['reference_id']:r for r in prepared}
pairs=[dict(**r,candidate_files=lookup[r['reference_id']]['candidates']) for r in results['pairs'] if r['method']=='custom' and r['metric']=='basic_ntokens' and r['ape'] is not None]
audit=dict(purpose='Check published supplement identity and expose discrepancies. These are source/arithmetic audits, not independent pipeline validation.',article='https://doi.org/10.3389/fpsyg.2021.661415',article_table1_checks=means,published_TTR_precision_matches=sum(r['within_published_precision'] for r in ratio_checks),published_TTR_rows=len(rows),largest_token_discrepancies=sorted(pairs,key=lambda r:r['ape'],reverse=True)[:10],exact_token_matches_by_method={m:sum(r['within_published_precision'] is True for r in results['pairs'] if r['method']==m and r['metric']=='basic_ntokens') for m in ['custom','taaled_current']},scope='The metadata join is documented but the original indexN.txt cleaned texts, original source/tier mapping, and historical GUI configuration remain unavailable. Do not infer the cause from errors, remap files to minimize error, or describe these as exact same-input reproduction.',supplement_listing='osf_listing.json: only TAALED_results.csv, TAALES_results.csv and LDLS_Analysis.R were present in the public OSF root.',unchanged_results_sha256=hashlib.sha256((ROOT/'outputs/results/lucid_validation/results.json').read_bytes()).hexdigest())
(evidence/'source_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print('Article means within 0.01 word:',all(x['within_0_01_word'] for x in means),'TTR consistency:',audit['published_TTR_precision_matches'],'/',len(rows))
