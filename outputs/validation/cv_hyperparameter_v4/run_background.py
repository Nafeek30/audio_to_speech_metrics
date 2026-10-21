"""Durable single-worker CV hyperparameter search, checks and report runner."""
from datetime import datetime,timezone
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).absolute().parents[3]
HERE=ROOT/'outputs/validation/cv_hyperparameter_v4'
OUT=ROOT/'outputs/results/cv_hyperparameter_v4'
STATUS=HERE/'job_status.json'
PYTHON=str(ROOT/'.venv/bin/python')


def now():return datetime.now(timezone.utc).isoformat()
def save(value):
 temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATUS)
def progress():
 protocol=json.loads((HERE/'protocol.json').read_text());result={}
 for config in protocol['recipes']:
  result[config]={}
  for name in [f'fold_{i}' for i in range(1,6)]+['final']:
   folder=OUT/config/name;entry=dict(selection_frozen=(folder/'selection.json').exists(),scored=(folder/('test_results.json' if name=='final' else 'cv_results.json')).exists())
   for filename in ['history.json','checkpoint_status.json']:
    try:
     value=json.loads((folder/filename).read_text())
     if filename=='history.json' and value:entry['latest_epoch']=value[-1]
     elif filename=='checkpoint_status.json':entry['checkpoint']=value
    except (FileNotFoundError,json.JSONDecodeError):pass
   result[config][name]=entry
 return result


def run():
 if (HERE/'user_stop.json').exists():
  print('User stopped this study; do not restart without a new request.',flush=True);return
 lock=(HERE/'run.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('CV supervisor already running',flush=True);return
 lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
 if STATUS.exists():
  previous=json.loads(STATUS.read_text())
  if previous.get('phase')=='complete':return
  if previous.get('worker_pid'):
   try:os.kill(previous['worker_pid'],0)
   except ProcessLookupError:pass
   else:
    state=subprocess.run(['ps','-p',str(previous['worker_pid']),'-o','stat='],capture_output=True,text=True).stdout.strip()
    if state and not state.startswith('Z'):raise RuntimeError('Prior worker still alive; refuse duplicate')
 state=dict(supervisor_pid=os.getpid(),started_utc=now(),phase='starting',progress=progress(),stages=[]);save(state)
 env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='/private/tmp/speech_pipeline_matplotlib')
 stages=[('cv_training_selection_and_test',['-m','speech_pipeline.research.cv_hyperparameter','run']),
  ('runtime_parity_and_activation',['-m','speech_pipeline.research.cv_hyperparameter_runtime','--activate']),
  ('combined_report',['-m','speech_pipeline.research.transcript_report']),
  ('cohort_verification',['-m','speech_pipeline.cohorts','verify']),
  ('regression_tests',['-m','unittest','discover','-s','tests','-v'])]
 for name,args in stages:
  state.update(phase=name,heartbeat_utc=now());save(state)
  with (HERE/(name+'.log')).open('a') as log:
   log.write('\nSTAGE START '+now()+'\n')
   worker=subprocess.Popen([PYTHON,'-u',*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   state.update(worker_pid=worker.pid,command=[PYTHON,'-u',*args]);save(state)
   while worker.poll() is None:
    state.update(heartbeat_utc=now(),progress=progress());save(state);time.sleep(10)
  state['stages'].append(dict(name=name,exit_code=worker.returncode,finished_utc=now()))
  if worker.returncode:
   state.update(phase='failed',failed_stage=name,finished_utc=now(),progress=progress());save(state);return
 required=[OUT/'cv_results.json',OUT/'cv_selection.json',OUT/'REPORT.md',HERE/'final_integrity_audit.json',HERE/'runtime_parity.json',ROOT/'outputs/results/transcript_validation/REPORT.md']
 state.update(phase='complete',finished_utc=now(),progress=progress(),artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in required});save(state)


if __name__=='__main__':
 try:run()
 except BaseException:
  error=traceback.format_exc();save(dict(phase='failed',supervisor_pid=os.getpid(),finished_utc=now(),traceback=error));raise
