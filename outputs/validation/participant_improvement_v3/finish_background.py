"""Finish audited runtime/report artifacts if the conversation is interrupted."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).absolute().parents[3]
FOLDER=ROOT/'outputs/validation/participant_improvement_v3'
STATUS=FOLDER/'finish_status.json'
PARENT=ROOT/'outputs/validation/participant_cv_v1/job_status.json'
PYTHON=str(ROOT/'.venv/bin/python')


def now():return datetime.now(timezone.utc).isoformat()
def save(value):
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATUS)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    lock=(FOLDER/'finish.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if STATUS.exists() and json.loads(STATUS.read_text()).get('phase')=='complete':return
    state=dict(pid=os.getpid(),started_utc=now(),phase='waiting_for_training_and_selection');save(state)
    while True:
        parent=json.loads(PARENT.read_text())
        if parent['phase']=='complete':break
        if parent['phase'] in ['failed','supervisor_failed']:
            state.update(phase='training_failed',parent_state=parent,finished_utc=now());save(state);return
        state.update(heartbeat_utc=now());save(state);time.sleep(10)
    stages=[('runtime_parity_and_deployment',['-m','speech_pipeline.research.participant_improvement_refined_runtime','--activate']),
        ('scientific_figure',['-m','speech_pipeline.research.participant_improvement_refined_figures']),
        ('combined_report',['-m','speech_pipeline.research.transcript_report']),
        ('final_cohort_verification',['-m','speech_pipeline.cohorts','verify']),
        ('regression_tests',['-m','unittest','discover','-s','tests','-v'])]
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='/private/tmp/speech_pipeline_matplotlib')
    state['stages']=[]
    for name,args in stages:
        state.update(phase=name,heartbeat_utc=now());save(state)
        with (FOLDER/(name+'.log')).open('a') as log:
            p=subprocess.Popen([PYTHON,'-u',*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=p.pid;save(state)
            while p.poll() is None:
                state.update(heartbeat_utc=now());save(state);time.sleep(5)
        state['stages'].append(dict(name=name,exit_code=p.returncode,finished_utc=now()))
        if p.returncode:
            state.update(phase='failed',failed_stage=name,finished_utc=now());save(state);return
    required=[ROOT/'outputs/results/participant_improvement_v3/REPORT.md',
        ROOT/'outputs/results/participant_improvement_v3/test_results.json',
        ROOT/'outputs/results/participant_improvement_v3/test_comparison.png',
        ROOT/'outputs/results/transcript_validation/REPORT.md',FOLDER/'runtime_parity.json',FOLDER/'final_integrity_audit.json']
    state.update(phase='complete',finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in required});save(state)


if __name__=='__main__':
    try:run()
    except BaseException:
        save(dict(phase='failed',pid=os.getpid(),finished_utc=now(),traceback=traceback.format_exc()));raise
