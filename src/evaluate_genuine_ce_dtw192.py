"""Evaluate the fixed genuine-only CE checkpoint at DTW192; never train.

Retain reports/scores and delete only this job's marked temporary export after
the original embedding scorer has been reproduced and all results validated.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from run_resolution_genuine_sequence import atomic_json, fingerprint
from scoped_export_cleanup import (
    sha, inside, export_path, prepare_export, validated_report,
    describe_grid, cleanup_export,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = 'dtw192-genuinece-s1111-20260916-v1-genuine-ce'
TAG = 'genuine-ce-dtw192-s1111-20260916-v1'
EXPECTED_WEIGHT = '6353f9102a6620ae8d1d00cddf1a8c2861610e717f5ec2bcc6456692bab59308'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('physical GPU 1 only')
    os.chdir(ROOT)
    checkpoint = inside(ROOT, ROOT/'models'/TRAIN/'1111/best_train_loss.pt', 'models')
    if sha(checkpoint) != EXPECTED_WEIGHT:
        raise RuntimeError('genuine CE best checkpoint differs from the verified epoch99 weight')
    previous = ROOT/'results'/TRAIN/'matched-local-seed1111'
    reference = validated_report(ROOT, previous, 96)
    if reference['source_sha256']['verify_fusion_matched.py'] != sha(ROOT/'src/verify_fusion_matched.py'):
        raise RuntimeError('evaluator differs from the verified DTW96 run')
    old_status = json.loads((ROOT/'results'/TRAIN/'status.json').read_text())
    if old_status['phase'] != 'completed' or old_status['best_epoch'] != 99:
        raise RuntimeError('training status does not match the fixed checkpoint')
    out = inside(ROOT, ROOT/'results'/TAG, 'results')
    temporary = 'tmp-'+TAG+'-local'
    if out.exists() or export_path(ROOT, temporary).exists():
        raise RuntimeError('output already exists; inspect before retrying (no overwrite)')
    fixed = fingerprint(ROOT)
    fixed[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
    fixed[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    fixed[str((previous/'summary.json').relative_to(ROOT))] = sha(previous/'summary.json')
    out.mkdir(exist_ok=False)
    state = dict(phase='initializing',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                 run_tag=TAG,training_run=TRAIN,checkpoint=str(checkpoint),best_epoch=99,
                 length=192,band=.25,alpha=.5,physical_gpu=1,no_retraining=True,fingerprint=fixed)

    def status(**changes):
        state.update(changes)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out/'status.json',state)

    def check_fixed():
        for name, digest in fixed.items():
            if sha(ROOT/name) != digest:
                raise RuntimeError('source/checkpoint/reference changed: '+name)

    def execute(command,phase):
        check_fixed()
        with (out/'evaluation.log').open('a') as log:
            log.write('\n'+json.dumps(command)+'\n');log.flush()
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(phase=phase,child_pid=child.pid)
            rc = child.wait()
        if rc:
            raise RuntimeError('child exit %d; see evaluation.log'%rc)
        status(child_pid=None)

    try:
        status()
        prepare_export(ROOT,temporary,checkpoint)
        execute([sys.executable,'-u','-B','src/evaluate_stylus.py',
                 '--model-name','integer-tcn-mamba2-interval','--data-source','real',
                 '--data-root','data_real_forgery/deepsigndb_stylus_v1','--eval-datasets','deepsigndb',
                 '--checkpoint',str(checkpoint),'--run-tag',TRAIN,'--export-tag',temporary,
                 '--seed','1111','--epoch','BestTrainLoss','--export-seq-feats'], 'exporting_local_features')
        report_dir = out/'matched-local-seed1111'
        execute([sys.executable,'-u','-B','src/verify_fusion_matched.py',
                 '--baseline-export',TRAIN+'-best-train-loss','--local-export',temporary,
                 '--baseline-results',str(ROOT/'results'/TRAIN/'embedding_baseline_for_audit'),
                 '--output-dir',str(report_dir),'--data-root','data_real_forgery/deepsigndb_stylus_v1',
                 '--length','192','--band','0.25','--seed','1111'], 'verifying_dtw192')
        check_fixed()
        report = validated_report(ROOT,report_dir,192)
        for n in ('1','4'):
            if report['results'][n]['embedding']['eer_percent'] != reference['results'][n]['embedding']['eer_percent']:
                raise RuntimeError('embedding EER differs from the same checkpoint at DTW96')
        describe_grid(ROOT,report_dir,192)
        comparison = dict(
            genuine_ce_dtw96={n:reference['results'][n] for n in ('1','4')},
            genuine_ce_dtw192={n:report['results'][n] for n in ('1','4')},
            dtw_delta_pp={n:report['results'][n]['dtw']['eer_percent']-reference['results'][n]['dtw']['eer_percent'] for n in ('1','4')},
            note='Same epoch99 checkpoint, original scorer and pooled EER; only the local interpolation grid changes. Single seed exploratory evaluation.')
        atomic_json(out/'comparison.json',comparison)
        status(phase='cleaning_owned_temporary_export')
        cleanup = cleanup_export(ROOT,temporary,report_dir,192,out/'temporary_export_cleanup.json')
        status(phase='completed',child_pid=None,finished_at=datetime.now(timezone.utc).isoformat(),
               dtw_eer={n:report['results'][n]['dtw']['eer_percent'] for n in ('1','4')},
               fusion_eer={n:report['results'][n]['fusion']['eer_percent'] for n in ('1','4')},
               temporary_export_removed=cleanup['completed'],freed_bytes=cleanup['freed_bytes'])
        print(json.dumps(state,indent=2),flush=True)
    except BaseException as exc:
        status(phase='failed',error=str(exc))
        raise


if __name__ == '__main__':
    main()
