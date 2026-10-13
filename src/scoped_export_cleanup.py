"""Own and clean only explicitly marked derived exports INSIDE one project.

No recursive deletion, no outside-project targets, no symlinks, no checkpoint,
dataset or report removal. Cleanup requires complete audited score artifacts.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


SUBSETS=('mcyt','bsid','bsds2','ebio1','ebio2')
MARKER='_temporary_export_owner.json'
NAMES={p+'_epochBestTrainLoss.npy' for p in ('feats','labels','inds','seq_feat','seq_mask')} | {'real_data_provenance.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def inside(root,path,section):
    root=Path(root).resolve()
    path=Path(path).absolute()
    if path.resolve()!=path or path.is_symlink():
        raise ValueError('symlink or redirected path refused: '+str(path))
    parent=root/section
    if path==parent or parent not in path.parents:
        raise ValueError('outside allowed project section: '+str(path))
    return path


def export_path(root,tag):
    if not re.fullmatch(r'tmp-[A-Za-z0-9][A-Za-z0-9_.-]*',tag):
        raise ValueError('cleanup is limited to tmp- export tags')
    return inside(root,Path(root).resolve()/'exps'/tag,'exps')


def prepare_export(root,tag,checkpoint):
    target=export_path(root,tag)
    checkpoint=inside(root,checkpoint,'models')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    identity=dict(kind='derived-local-export-v1',tag=tag,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    if target.exists():
        marker=inside(root,target/MARKER,'exps')
        if not marker.is_file() or json.loads(marker.read_text())!=identity:
            raise ValueError('refusing an existing unowned/different export')
    else:
        if shutil.disk_usage(Path(root)).free < 45*1024**3:
            raise RuntimeError('at least 45 GiB free required before full local feature export')
        target.mkdir(parents=True,exist_ok=False)
        with (target/MARKER).open('x') as f: json.dump(identity,f,indent=2)
    return target


def validated_report(root,report_dir,length):
    directory=inside(root,report_dir,'results')
    file=inside(root,directory/'summary.json','results')
    r=json.loads(file.read_text())
    if r.get('length')!=length or r.get('band')!=.25 or r.get('alpha')!=.5:
        raise ValueError('unexpected evaluation settings')
    for n in ('1','4'):
        a=r['audit'][n]
        if not a.get('passed') or not a.get('per_query_scores_match'):
            raise ValueError('embedding audit incomplete')
        for branch in ('embedding','pointwise','dtw','fusion'):
            value=r['results'][n][branch]['eer_percent']
            if not math.isfinite(value) or not 0<=value<=100:
                raise ValueError('incomplete/nonfinite result')
        scores=inside(root,directory/('all_scores_%svs1.npz'%n),'results')
        if not scores.is_file() or scores.stat().st_size==0:
            raise ValueError('per-query evidence missing')
    return r


def describe_grid(root,report_dir,length):
    # Fix an old hard-coded descriptive string, without changing any score/EER
    # or the original evaluator source. Numeric `length` already is correct.
    validated_report(root,report_dir,length)
    for filename in ('summary.json','embedding_audit.json'):
        p=inside(root,Path(report_dir)/filename,'results')
        r=json.loads(p.read_text())
        if r['length']!=length: raise ValueError('wrong report length')
        r['local_preprocessing']='%d-grid linear interpolation then per-vector L2; cosine cost'%length
        r['description_helper_sha256']=sha(__file__)
        with p.open('w') as f: json.dump(r,f,indent=2)


def cleanup_export(root,tag,report_dir,length,cleanup_report):
    root=Path(root).resolve()
    target=export_path(root,tag)
    log=inside(root,cleanup_report,'results')
    validated_report(root,report_dir,length)
    if not target.exists():
        if log.is_file():
            previous=json.loads(log.read_text())
            if previous.get('export')==str(target) and previous.get('completed'):
                return previous
        raise ValueError('no owned export or completed cleanup record')
    marker=inside(root,target/MARKER,'exps')
    owner=json.loads(marker.read_text())
    if owner.get('kind')!='derived-local-export-v1' or owner.get('tag')!=tag:
        raise ValueError('invalid ownership marker')
    checkpoint=inside(root,owner['checkpoint'],'models')
    if sha(checkpoint)!=owner['checkpoint_sha256']:
        raise ValueError('checkpoint changed; cannot guarantee regeneration')
    files=[];directories=[]
    for p in target.rglob('*'):
        inside(root,p,'exps')
        rel=p.relative_to(target)
        if p.is_dir():
            if rel.parts[0]!='seed1111' or len(rel.parts)>2 or (len(rel.parts)==2 and rel.parts[1] not in SUBSETS):
                raise ValueError('unexpected export subdirectory')
            directories.append(p)
        elif p!=marker:
            if len(rel.parts)!=3 or rel.parts[0]!='seed1111' or rel.parts[1] not in SUBSETS or rel.name not in NAMES:
                raise ValueError('unexpected file; refusing cleanup: '+str(p))
            files.append(dict(path=str(p),bytes=p.stat().st_size))
    record=dict(export=str(target),checkpoint=str(checkpoint),checkpoint_sha256=owner['checkpoint_sha256'],
                report=str(report_dir),report_sha256=sha(Path(report_dir)/'summary.json'),
                files=files,freed_bytes=0,completed=False)
    if log.exists():
        old=json.loads(log.read_text())
        if old.get('export')!=str(target): raise ValueError('cleanup log belongs to another target')
    with log.open('w') as f: json.dump(record,f,indent=2)
    # No broad rm/rmtree: only the individually validated derived files.
    for item in sorted(files,key=lambda x:x['bytes'],reverse=True):
        p=inside(root,item['path'],'exps')
        if p.stat().st_size!=item['bytes']: raise ValueError('file changed during cleanup')
        p.unlink();record['freed_bytes']+=item['bytes']
    for d in sorted(directories,key=lambda x:len(x.parts),reverse=True):
        inside(root,d,'exps').rmdir()
    inside(root,marker,'exps').unlink()
    inside(root,target,'exps').rmdir()
    if sha(checkpoint)!=owner['checkpoint_sha256']: raise RuntimeError('checkpoint integrity failure')
    record['completed']=True
    with log.open('w') as f: json.dump(record,f,indent=2)
    return record
