"""Detached supervisor for the registered CV job; no training choices here."""
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=ROOT/'outputs/validation/participant_cv_v1'
OUT=ROOT/'outputs/results/participant_cv_v1'
STATUS=EVIDENCE/'job_status.json'
LOG=EVIDENCE/'training.log'


def now():return datetime.now(timezone.utc).isoformat()
def write(value):
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATUS)


def progress():
    result={}
    for name in ['fold_1','fold_2','fold_3','fold_4','fold_5','final']:
        folder=OUT/name;entry=dict(selection_frozen=(folder/'selection.json').exists(),evaluated=(folder/'outer_results.json').exists())
        for filename in ['history.json','checkpoint_status.json']:
            try:
                value=json.loads((folder/filename).read_text())
                if filename=='history.json' and value:entry['latest_epoch']=value[-1]
                elif filename=='checkpoint_status.json':entry['checkpoint']=value
            except (FileNotFoundError,json.JSONDecodeError):pass
        result[name]=entry
    return result


def main():
    lock=(EVIDENCE/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('A detached supervisor is already running; inspect job_status.json.',flush=True)
        return
    lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
    if STATUS.exists():
        previous=json.loads(STATUS.read_text())
        if previous.get('phase')=='running' and previous.get('worker_pid'):
            try:os.kill(previous['worker_pid'],0)
            except ProcessLookupError:pass
            else:
                print('The saved training worker is still alive; refusing a duplicate run.',flush=True)
                return
    legacy=int(sys.argv[1]) if len(sys.argv)>1 else None
    state=dict(supervisor_pid=os.getpid(),started_utc=now(),phase='waiting_for_existing_worker' if legacy else 'starting',
        legacy_worker_pid=legacy,log=str(LOG),command=[str(ROOT/'.venv/bin/python'),'-u','-m','speech_pipeline.research.participant_improvement_refined','run'])
    write(state)
    if legacy:
        while True:
            try:os.kill(legacy,0)
            except ProcessLookupError:break
            state_text=subprocess.run(['ps','-p',str(legacy),'-o','stat='],capture_output=True,text=True).stdout.strip()
            if not state_text or state_text.startswith('Z'):break
            old_log=Path('/private/tmp/participant_cv_v1.log')
            if old_log.exists():shutil.copy2(old_log,EVIDENCE/'foreground_initial.log')
            state.update(heartbeat_utc=now(),progress=progress());write(state);time.sleep(10)
        state['legacy_worker_finished_utc']=now()
        old_log=Path('/private/tmp/participant_cv_v1.log')
        if old_log.exists():shutil.copy2(old_log,EVIDENCE/'foreground_initial.log')
        # If the foreground worker was stopped before freezing its first fold,
        # preserve its incomplete artifacts and restart that fold from the same
        # registered original initialization. Never reuse partial task weights.
        old=OUT/'fold_1'
        if old.exists() and not (old/'selection.json').exists() and not (old/'training_state.pt').exists():
            destination=OUT/('legacy_interrupted_fold_1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
            shutil.move(str(old),str(destination))
            state['legacy_recovery']=dict(archived=str(destination),reason='Foreground worker ended without optimizer/RNG checkpoint; restart same fixed fold/seed from original pretrained weights')
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with LOG.open('a',buffering=1) as log:
        log.write('\nDETACHED RUNNER START '+now()+'\n')
        worker=subprocess.Popen(state['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(worker_pid=worker.pid,phase='running');write(state)
        while worker.poll() is None:
            state.update(heartbeat_utc=now(),progress=progress());write(state);time.sleep(10)
        state.update(exit_code=worker.returncode,finished_utc=now(),progress=progress())
        expected=[OUT/f'fold_{i}/outer_results.json' for i in range(1,6)]+[OUT/'cross_validation_results.json',OUT/'REPORT.md']
        expected += [OUT/'final/selection.json',EVIDENCE/'development_complete.json',
            ROOT/'outputs/results/participant_improvement_v3/test_results.json',
            ROOT/'outputs/results/participant_improvement_v3/REPORT.md',
            ROOT/'outputs/validation/participant_improvement_v3/final_integrity_audit.json']
        state['all_expected_outputs_present']=all(p.exists() for p in expected)
        state['phase']='complete' if worker.returncode==0 and state['all_expected_outputs_present'] else 'failed'
        write(state)


if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc()
        write(dict(phase='supervisor_failed',supervisor_pid=os.getpid(),finished_utc=now(),error=error))
        with LOG.open('a') as log:log.write(error)
        raise
