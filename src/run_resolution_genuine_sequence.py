"""First test fixed-weight DTW192; then train genuine-only CE independently.

No stacking of the two changes: CE's main evaluation remains DTW96. Retain
all checkpoints and score reports, clean only owned project-local exports.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scoped_export_cleanup import sha,prepare_export,cleanup_export,validated_report,describe_grid

ROOT=Path(__file__).resolve().parents[1]
BASELINE='interval-mamba-deepsign-central-loss-s1111-server-v1'


def atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def fingerprint(root):
    names=['src/'+n for n in ('run_resolution_genuine_sequence.py','run_genuine_ce_experiment.py',
        'scoped_export_cleanup.py','verify_fusion_matched.py','evaluation_utils.py','run_integer_tcn_experiment.py',
        'main.py','train_integer_tcn.py','integer_tcn_config.py','model_factory.py','network_integer_tcn.py',
        'network_integer_tcn_mamba.py','network_interval_mamba.py','learnable_interval_spike.py','mamba2_torch.py',
        'dataset/datasetRealForgery.py','real_forgery_data.py','deepsigndb_data.py','optimAP.py','evaluate_stylus.py',
        'verifier_deepsign_central.py','verifier_stylus_embedding.py','verifier_stylus_snn_tvs.py')]
    names+=['configs/mamba_genuine_only_ce.json',
            'models/'+BASELINE+'/1111/best_train_loss.pt',
            'results/'+BASELINE+'/matched-audit-20260915-v1/summary.json']
    return {name:sha(root/name) for name in names}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-tag',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.sequence_tag): p.error('unsafe tag')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise RuntimeError('physical GPU1 only')
    os.chdir(ROOT)
    baseline_dir=ROOT/'results'/BASELINE
    original=validated_report(ROOT,baseline_dir/'matched-audit-20260915-v1',96)
    if original['source_sha256']['verify_fusion_matched.py']!=sha(ROOT/'src/verify_fusion_matched.py'):
        raise RuntimeError('matched evaluator differs from audited baseline')
    fixed=fingerprint(ROOT)
    out=ROOT/'results'/args.sequence_tag
    if args.resume:
        state=json.loads((out/'sequence_status.json').read_text())
        if state['fingerprint']!=fixed: raise RuntimeError('source/baseline changed since sequence started')
    else:
        out.mkdir(parents=True,exist_ok=False)
        state=dict(phase='initializing',sequence_tag=args.sequence_tag,started_at=datetime.now(timezone.utc).isoformat(),
                   completed_stages=[],fingerprint=fixed,independent_experiments=True,
                   baseline_dtw={n:original['results'][n]['dtw']['eer_percent'] for n in ('1','4')})
    def status(**kw):
        state.update(kw);state['updated_at']=datetime.now(timezone.utc).isoformat();state['pid']=os.getpid()
        atomic_json(out/'sequence_status.json',state)
    def execute(command,logname):
        with (out/logname).open('a') as log:
            log.write('\n'+json.dumps(command)+'\n');log.flush()
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(child_pid=child.pid)
            rc=child.wait()
        if rc: raise RuntimeError('child failed (%d); see %s'%(rc,logname))
        status(child_pid=None)
    def summary_scores(report):
        values={n:report['results'][n]['dtw']['eer_percent'] for n in ('1','4')}
        return dict(dtw_eer=values,delta_pp={n:values[n]-state['baseline_dtw'][n] for n in values},
                    both_improved=all(values[n]<state['baseline_dtw'][n]-1e-6 for n in values))
    try:
        if 'resolution192' not in state['completed_stages']:
            status(phase='resolution192')
            target=out/'resolution192'
            local_tag=BASELINE+'-dtw-local'
            temporary='tmp-'+args.sequence_tag+'-resolution192'
            owned=False
            cache=ROOT/'exps'/local_tag/'seed1111'
            present=all((cache/sub/'seq_feat_epochBestTrainLoss.npy').is_file() for sub in ('mcyt','bsid','bsds2','ebio1','ebio2'))
            if not (target/'summary.json').exists():
                if not present:
                    checkpoint=ROOT/'models'/BASELINE/'1111/best_train_loss.pt'
                    prepare_export(ROOT,temporary,checkpoint);local_tag=temporary;owned=True
                    execute([sys.executable,'-u','-B','src/evaluate_stylus.py','--model-name','integer-tcn-mamba2-interval',
                        '--data-source','real','--data-root','data_real_forgery/deepsigndb_stylus_v1','--eval-datasets','deepsigndb',
                        '--checkpoint',str(checkpoint),'--run-tag',BASELINE,'--export-tag',local_tag,'--seed','1111',
                        '--epoch','BestTrainLoss','--export-seq-feats'],'resolution192.log')
                execute([sys.executable,'-u','-B','src/verify_fusion_matched.py','--baseline-export',BASELINE+'-best-train-loss',
                    '--local-export',local_tag,'--baseline-results',str(baseline_dir),'--output-dir',str(target),
                    '--length','192','--band','0.25','--seed','1111'],'resolution192.log')
            report=validated_report(ROOT,target,192)
            describe_grid(ROOT,target,192)
            if owned or (ROOT/'exps'/temporary).exists():
                cleanup_export(ROOT,temporary,target,192,out/'resolution192_cleanup.json')
            else:
                atomic_json(out/'resolution192_cleanup.json',dict(reused_existing_cache=str(cache),deleted_files=[],
                                                                 note='original cache not owned by this run; preserved'))
            state['resolution192']=summary_scores(report)
            state['completed_stages'].append('resolution192')
            status(phase='resolution192_completed')
        if fingerprint(ROOT)!=fixed: raise RuntimeError('source changed before genuine CE stage')
        if 'genuine_ce' not in state['completed_stages']:
            tag=args.sequence_tag+'-genuine-ce'
            status(phase='genuine_ce_training_and_evaluating',run_tag=tag)
            command=[sys.executable,'-u','-B','src/run_genuine_ce_experiment.py','--config','configs/mamba_genuine_only_ce.json',
                     '--run-tag',tag,'--seeds','1111','--device','cuda','--microbatch-tasks','1','--cpu-threads','4']
            run_result=ROOT/'results'/tag
            if (run_result/'run_config.json').exists():
                if not args.resume: raise RuntimeError('stage already exists; refusing duplicate')
                command.append('--resume')
            execute(command,'genuine_ce.log')
            if json.loads((run_result/'status.json').read_text())['phase']!='completed':
                raise RuntimeError('genuine CE stage incomplete')
            report=validated_report(ROOT,run_result/'matched-local-seed1111',96)
            cleanup=json.loads((run_result/'temporary_export_cleanup.json').read_text())
            if not cleanup['completed']: raise RuntimeError('temporary feature cleanup incomplete')
            state['genuine_ce']=summary_scores(report)
            state['completed_stages'].append('genuine_ce')
        status(phase='completed',finished_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps({k:state[k] for k in ('baseline_dtw','resolution192','genuine_ce')},indent=2),flush=True)
    except BaseException as exc:
        status(phase='failed',error=str(exc));raise


if __name__=='__main__': main()
