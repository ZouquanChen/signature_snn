"""Independent AP(all episode samples) + CE(genuine anchor/positives only).

No model change; CE is averaged over genuine samples (not diluted by fake
sample count). Uses original experiment recovery with process-local callbacks.
Temporary local exports are deleted ONLY after audited evaluation completes.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import run_integer_tcn_experiment as base
from scoped_export_cleanup import prepare_export,cleanup_export,validated_report,describe_grid


IDENTITY_CE='genuine_only: anchor plus genuine positives; mean over genuine samples'


def genuine_ce(model,logits,labels):
    step=1+model.n_shot_g+model.n_shot_f
    if len(logits)%step or len(labels)!=len(logits):
        raise ValueError('invalid episode layout for genuine-only CE')
    mask=(torch.arange(len(logits),device=logits.device)%step)<1+model.n_shot_g
    return model.smoothCEloss(logits[mask],labels[mask],eps=.1)


def backward_batch(model,batch,device,microbatch_tasks=1):
    signals,lengths,labels=batch
    step=1+model.n_shot_g+model.n_shot_f
    if len(signals)!=model.n_task*step or microbatch_tasks<1: raise ValueError('invalid episode batch')
    metrics=dict(total_loss=0.,ap_loss=0.,ce_loss=0.,train_map=0.)
    for start in range(0,len(signals),microbatch_tasks*step):
        end=min(len(signals),start+microbatch_tasks*step);tasks=(end-start)//step
        steps=max(int(np.max(lengths[start:end])),model.n_in+1)
        x=torch.from_numpy(signals[start:end,:steps]).to(device)
        lens=torch.as_tensor(lengths[start:end],device=device)
        target=torch.as_tensor(labels[start:end],device=device,dtype=torch.long)
        embedding,logits=model(x,lens=lens)
        ap,map_value=model.APLoss_DLM(embedding,n_task=tasks)
        ce=genuine_ce(model,logits,target)
        loss=ap+ce
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite AP + genuine-only CE')
        fraction=tasks/model.n_task
        (loss*fraction).backward()
        for k,v in (('total_loss',loss),('ap_loss',ap),('ce_loss',ce),('train_map',map_value)):
            metrics[k]+=float(v.detach())*fraction
    return metrics


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    job=base.build_parser().parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or job.seeds!=[1111]:
        raise ValueError('this standalone trial requires physical GPU1 and seed1111')
    config=json.loads(job.config.read_text())
    a=config['architecture']
    if (config.get('identity_ce')!='genuine_only' or a.get('interval_tcn',False)
            or a.get('interval_sites')!='both' or a.get('interval_input_gradient')!='normalized-triangle'
            or 'local_alignment' in config):
        raise ValueError('this experiment changes only the CE supervision subset')
    source=Path(__file__).resolve().parent
    names=(Path(__file__).name,'scoped_export_cleanup.py','verify_fusion_matched.py','evaluation_utils.py')
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    old_parser,old_backward,old_eval=base.train_integer_tcn.build_parser,base.backward_batch,base.evaluate_seed
    old_json,old_selection=base.atomic_json,base.SELECTION
    def parser(path=base.train_integer_tcn.DEFAULT_CONFIG):
        p=old_parser(path)
        p.set_defaults(identity_ce=IDENTITY_CE,genuine_ce_source_sha256=hashes)
        return p
    def save(path,value):
        if Path(path).name=='training_config.json': value=dict(value,identity_ce=IDENTITY_CE)
        return old_json(path,value)
    def evaluate(args,job,log_dir,result_dir,logger,status):
        for n,h in hashes.items():
            if hashlib.sha256((source/n).read_bytes()).hexdigest()!=h: raise RuntimeError('source changed: '+n)
        outcomes=old_eval(args,job,log_dir,result_dir,logger,status)
        target=result_dir/'matched-local-seed1111'
        temporary='tmp-'+job.run_tag+'-local'
        cleanup_log=result_dir/'temporary_export_cleanup.json'
        if (target/'summary.json').exists():
            validated_report(base.PROJECT,target,96)
            cleanup_export(base.PROJECT,temporary,target,96,cleanup_log)
            return outcomes
        checkpoint=base.PROJECT/'models'/job.run_tag/str(args.seed)/'best_train_loss.pt'
        prepare_export(base.PROJECT,temporary,checkpoint)
        logfile=log_dir/'seed1111-local-evaluation.log'
        status(dict(phase='exporting_temporary_local_features'))
        base.child_command([sys.executable,'-u','-B',str(source/'evaluate_stylus.py'),
            '--model-name',args.model_name,'--data-source','real','--data-root',args.data_root,
            '--eval-datasets','deepsigndb','--checkpoint',str(checkpoint),'--run-tag',job.run_tag,
            '--export-tag',temporary,'--seed','1111','--epoch','BestTrainLoss','--export-seq-feats'],logfile,logger)
        audit=result_dir/'embedding_baseline_for_audit';audit.mkdir(exist_ok=True)
        old_json(audit/'summary.json',{str(r)+'vs1':dict(mean_eer_percent=outcomes[str(r)]['mean_eer_percent']) for r in (1,4)})
        status(dict(phase='verifying_matched_local_dtw'))
        base.child_command([sys.executable,'-u','-B',str(source/'verify_fusion_matched.py'),
            '--baseline-export',job.run_tag+'-best-train-loss','--local-export',temporary,
            '--baseline-results',str(audit),'--output-dir',str(target),'--data-root',args.data_root,
            '--length','96','--seed','1111'],logfile,logger)
        describe_grid(base.PROJECT,target,96)
        status(dict(phase='cleaning_owned_temporary_features'))
        record=cleanup_export(base.PROJECT,temporary,target,96,cleanup_log)
        logger.info('Removed owned temporary export, %.3f GiB; weights/reports kept',record['freed_bytes']/1024**3)
        return outcomes
    base.train_integer_tcn.build_parser=parser;base.backward_batch=backward_batch
    base.evaluate_seed=evaluate;base.atomic_json=save
    base.SELECTION='minimum_sample_weighted_epoch_mean_AP_plus_genuine_only_CE'
    try: base.main(argv)
    finally:
        base.train_integer_tcn.build_parser=old_parser;base.backward_batch=old_backward
        base.evaluate_seed=old_eval;base.atomic_json=old_json;base.SELECTION=old_selection


if __name__=='__main__': main()
