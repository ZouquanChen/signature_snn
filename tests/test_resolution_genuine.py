import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import scoped_export_cleanup as cleanup
import run_genuine_ce_experiment as runner
from network_interval_mamba import IntervalTCNMamba2


def tiny():
    return IntervalTCNMamba2(12,3,width=8,embedding_dim=8,dilations=(1,2),
        mamba_layers=1,mamba_head_dim=4,mamba_state_dim=4,mamba_chunk_size=8,
        n_task=2,n_shot_g=1,n_shot_f=1)


def test_ce_excludes_forgery_logits_and_is_mean_over_genuine():
    model=tiny()
    x=torch.randn(6,3,requires_grad=True)
    labels=torch.tensor([0,0,0,1,1,1])
    actual=runner.genuine_ce(model,x,labels)
    ids=torch.tensor([0,1,3,4])
    expected=model.smoothCEloss(x[ids],labels[ids],eps=.1)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert x.grad[[2,5]].count_nonzero()==0
    assert x.grad[ids].abs().sum()>0
    model.n_shot_g=5;model.n_shot_f=10
    y=torch.randn(32,3,requires_grad=True);labels=torch.zeros(32,dtype=torch.long)
    loss=runner.genuine_ce(model,y,labels);loss.backward()
    assert (y.grad.abs().sum(1)>0).sum()==12


@pytest.mark.parametrize('checkpoint',[False,True])
def test_microbatch_and_checkpoint_equivalence(checkpoint):
    torch.set_num_threads(1);torch.manual_seed(8)
    a=tiny();b=copy.deepcopy(a);b.checkpoint_blocks=checkpoint
    batch=(np.random.default_rng(2).normal(size=(6,31,12)).astype('float32'),
           np.array([31,29,25,30,23,27]),np.array([0,0,0,1,1,1]))
    left=runner.backward_batch(a,batch,torch.device('cpu'),2)
    right=runner.backward_batch(b,batch,torch.device('cpu'),1)
    assert left==pytest.approx(right,abs=3e-6)
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4)


def test_selection_and_training_metadata_are_truthful(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    config=SRC.parent/'configs/mamba_genuine_only_ce.json'
    old_parser=runner.base.train_integer_tcn.build_parser
    old_backward=runner.base.backward_batch
    def fake_main(argv):
        a=runner.base.train_integer_tcn.build_parser(config).parse_args([])
        assert a.identity_ce==runner.IDENTITY_CE
        assert not a.interval_tcn and a.interval_sites=='both'
        assert runner.base.SELECTION.endswith('genuine_only_CE')
        f=tmp_path/'training_config.json'
        runner.base.atomic_json(f,{'identity_ce':'old incorrect label'})
        assert json.loads(f.read_text())['identity_ce']==runner.IDENTITY_CE
    monkeypatch.setattr(runner.base,'main',fake_main)
    runner.main(['--run-tag','test','--config',str(config),'--seeds','1111'])
    assert runner.base.train_integer_tcn.build_parser is old_parser
    assert runner.base.backward_batch is old_backward


@pytest.fixture
def owned(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir()
    weight=root/'models/run/1111/best_train_loss.pt';weight.parent.mkdir(parents=True);weight.write_bytes(b'preserved checkpoint')
    monkeypatch.setattr(cleanup.shutil,'disk_usage',lambda p:SimpleNamespace(free=60*1024**3))
    target=cleanup.prepare_export(root,'tmp-test',weight)
    d=target/'seed1111/mcyt';d.mkdir(parents=True)
    np.save(d/'seq_feat_epochBestTrainLoss.npy',np.ones((1,2,3)))
    (d/'real_data_provenance.json').write_text('{}')
    report=root/'results/run/matched';report.mkdir(parents=True)
    value=dict(length=96,band=.25,alpha=.5,
        audit={n:dict(passed=True,per_query_scores_match=True) for n in ('1','4')},
        results={n:{b:dict(eer_percent=2.) for b in ('embedding','pointwise','dtw','fusion')} for n in ('1','4')})
    (report/'summary.json').write_text(json.dumps(value))
    (report/'embedding_audit.json').write_text(json.dumps(value))
    for n in ('1','4'): np.savez(report/('all_scores_%svs1.npz'%n),scores=np.ones(3))
    return root,weight,target,report,root/'results/cleanup.json'


def test_cleanup_only_marked_derived_data_after_complete_report(owned,tmp_path):
    root,weight,target,report,log=owned
    outside=tmp_path/'outside.txt';outside.write_text('keep')
    data=root/'data_real_forgery';data.mkdir();(data/'raw.pkl').write_bytes(b'keep')
    before=cleanup.sha(weight)
    result=cleanup.cleanup_export(root,'tmp-test',report,96,log)
    assert result['completed'] and result['freed_bytes']>0 and not target.exists()
    assert cleanup.sha(weight)==before and outside.read_text()=='keep'
    assert (data/'raw.pkl').read_bytes()==b'keep'
    assert (report/'all_scores_1vs1.npz').exists()
    assert cleanup.cleanup_export(root,'tmp-test',report,96,log)==result


@pytest.mark.parametrize('tag',['../outside','tmp-../outside','original-baseline','tmp-x/../../models'])
def test_unsafe_or_nonowned_tags_rejected(owned,tag):
    root,weight,target,report,log=owned
    with pytest.raises(ValueError): cleanup.cleanup_export(root,tag,report,96,log)
    assert target.exists() and weight.exists()


def test_unknown_file_or_checkpoint_prevents_entire_cleanup(owned):
    root,weight,target,report,log=owned
    (target/'valuable.pt').write_bytes(b'keep')
    with pytest.raises(ValueError): cleanup.cleanup_export(root,'tmp-test',report,96,log)
    assert (target/'seed1111/mcyt/seq_feat_epochBestTrainLoss.npy').exists()


def test_outside_path_and_partial_report_are_refused(owned,tmp_path):
    root,weight,target,report,log=owned
    with pytest.raises(ValueError): cleanup.inside(root,tmp_path/'outside','exps')
    r=json.loads((report/'summary.json').read_text());del r['results']['4']
    (report/'summary.json').write_text(json.dumps(r))
    with pytest.raises((KeyError,ValueError)): cleanup.cleanup_export(root,'tmp-test',report,96,log)
    assert target.exists()


def test_checkpoint_mutation_prevents_cleanup(owned):
    root,weight,target,report,log=owned
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'): cleanup.cleanup_export(root,'tmp-test',report,96,log)
    assert target.exists()


def test_symlink_outside_never_followed(owned,tmp_path):
    root,weight,target,report,log=owned
    outside=tmp_path/'outside';outside.mkdir();keep=outside/'keep';keep.write_text('keep')
    try: (target/'escape').symlink_to(outside,target_is_directory=True)
    except OSError: pytest.skip('symlink creation unavailable')
    with pytest.raises(ValueError): cleanup.cleanup_export(root,'tmp-test',report,96,log)
    assert keep.read_text()=='keep'


def test_report_grid_description_only_no_score_changes(owned):
    root,weight,target,report,log=owned
    before=json.loads((report/'summary.json').read_text())
    cleanup.describe_grid(root,report,96)
    after=json.loads((report/'summary.json').read_text())
    assert before['results']==after['results'] and after['local_preprocessing'].startswith('96-grid')
