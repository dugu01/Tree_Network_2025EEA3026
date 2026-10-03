#!/usr/bin/env python3
"""Add a no-children output-refit control to an existing, unfinalized run.

Uses original root checkpoints and cached training features. Does not retrain
the CNN, change existing models, read test arrays, or replace core source files.
Only run on trusted checkpoints created by this project.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import read_model,save_model,write_metrics,source_digest
from data_features import save_json,digest_file
from reporting import build_report
from threadpoolctl import threadpool_limits
import argparse
import copy
import json
import csv
import shutil
import time
import numpy as np
import torch


def audit_run(run_dir,cache_override=None):
    run_dir=Path(run_dir).resolve()
    if not (run_dir/'cache_reference.json').exists() or not (run_dir/'root'/'model.pkl').exists():
        raise SystemExit(f'No trained root checkpoint/cache reference in {run_dir}. Complete the training command successfully before running this audit.')
    if (run_dir/'selection_locked.json').exists():
        raise RuntimeError('This run has already been finalized. Keep its test results fixed; use an unfinalized run for the additional control.')
    cache=Path(cache_override) if cache_override else Path(json.loads((run_dir/'cache_reference.json').read_text())['path'])
    if not cache.exists():raise FileNotFoundError('Feature cache is missing. Run this script on the computer with the original checkpoint and cache, or supply --cache-path.')
    meta=json.loads((cache/'metadata.json').read_text())
    manifest=json.loads((cache/'split_manifest.json').read_text())
    manifest_digest=digest_file(cache/'split_manifest.json')
    if manifest_digest!=meta['manifest_sha256']:raise ValueError('Split manifest has changed')
    ids=[{r['id'] for r in manifest[s]} for s in ['train','val','test']]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Overlapping split identifiers')
    checkpoint=read_model(run_dir/'root'/'model.pkl')
    if checkpoint['metadata']['manifest_sha256']!=manifest_digest:raise ValueError('Root checkpoint and cache belong to different splits')
    arrays={}
    for split in ['train','val']:
        for suffix in ['X','y']:
            path=cache/f'{split}_{suffix}.npy'
            if digest_file(path)!=meta['file_hashes'][path.name]:raise ValueError(f'Cache checksum mismatch: {path.name}')
        X=np.load(cache/f'{split}_X.npy');y=np.load(cache/f'{split}_y.npy')
        if len(y)!=len(manifest[split]):raise ValueError('Manifest and array lengths differ')
        if not np.array_equal(y,np.array([r['label'] for r in manifest[split]])):raise ValueError('Manifest and array labels differ')
        arrays[split]=((X-checkpoint['mean'])/checkpoint['std'],y)
    destination=run_dir/'root_refit';destination.mkdir(exist_ok=True)
    request={'control':'Original root MLP, frozen hidden layer, constrained logistic output refit; no children',
             'core_source_sha256':source_digest(),'addon_sha256':digest_file(__file__),
             'root_checkpoint_sha256':digest_file(run_dir/'root'/'model.pkl'),'manifest_sha256':manifest_digest}
    rp=destination/'control_request.json'
    if rp.exists() and json.loads(rp.read_text())!=request:raise ValueError('Control inputs changed; do not overwrite the earlier control')
    save_json(rp,request)
    if not (destination/'metrics.json').exists():
        start=time.perf_counter();model=copy.deepcopy(checkpoint['model']);X,y=arrays['train']
        targets=[model.classes[1]] if len(model.classes)==2 else model.classes
        for node,label in zip(model.roots,targets):
            if node.A is not None or node.B is not None:raise AssertionError('Expected a root-only checkpoint')
            local_y=(y==label).astype(np.int64);old=node.base_score(X)
            errors=int(np.sum((old>=0)!=local_y))
            node.stats['base_errors']=errors
            if node.model is not None:node._refit(X,local_y,old)
            node.stats['final_errors']=int(np.sum(node.predict(X)!=local_y))
            assert node.A is None and node.B is None
            assert node.stats['final_errors']<=errors
        save_model(destination/'model.pkl',model,checkpoint['mean'],checkpoint['std'],meta)
        metrics=write_metrics(destination,model,arrays,meta,vars(model.config),'root_refit',start)
        metrics['training_seconds_note']='Incremental output-refit and metric time only; original MLP training cost is separate.'
        metrics['control']=request['control'];save_json(destination/'metrics.json',metrics)
    print('Root refit control complete; original root/tree/pruned checkpoints preserved.')
    models={}
    for kind in ['root','root_refit','tree','pruned']:
        p=run_dir/kind/'model.pkl'
        if p.exists():models[kind]=read_model(p)['model']
    audit={'test_arrays_read':False,'split_id_overlap':False,'feature_checksums_verified':True,
           'feature_extraction_seconds':meta.get('seconds'),'feature_extraction_device':meta.get('device'),
           'counts':meta.get('counts'),'comparisons':{},'metrics_recomputed_from_checkpoints':{}}
    for split,(X,y) in arrays.items():
        predictions={k:m.predict(X) for k,m in models.items()}
        with open(run_dir/f'{split}_predictions_audit.csv','w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','label']+list(models))
            for i,target in enumerate(y):writer.writerow([manifest[split][i]['id'],int(target)]+[int(predictions[k][i]) for k in models])
        audit['metrics_recomputed_from_checkpoints'][split]={}
        for kind,pred in predictions.items():
            errors=int(np.sum(pred!=y));saved=json.loads((run_dir/kind/'metrics.json').read_text())
            if saved[split]['errors']!=errors:raise AssertionError(f'Saved accuracy differs from reloaded checkpoint: {kind}/{split}')
            audit['metrics_recomputed_from_checkpoints'][split][kind]={'errors':errors,'n':len(y),'accuracy':float(np.mean(pred==y))}
        comparisons={}
        for left,right in [('root','root_refit'),('root_refit','tree'),('tree','pruned')]:
            if left not in predictions or right not in predictions:continue
            a,b=predictions[left],predictions[right]
            comparisons[f'{left}_to_{right}']={'prediction_disagreements':int(np.sum(a!=b)),
                'corrected':int(np.sum((a!=y)&(b==y))),'spoiled':int(np.sum((a==y)&(b!=y)))}
        audit['comparisons'][split]=comparisons
    save_json(run_dir/'control_audit.json',audit)
    shutil.copy2(cache/'metadata.json',run_dir/'cache_metadata.json')
    shutil.copy2(cache/'split_manifest.json',run_dir/'split_manifest.json')
    if (cache/'excluded_images.json').exists():
        shutil.copy2(cache/'excluded_images.json',run_dir/'excluded_images.json')
    print(json.dumps(audit['metrics_recomputed_from_checkpoints'],indent=2))
    print('Prediction-level comparisons:',json.dumps(audit['comparisons'],indent=2))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='results/cifar10/pilot/baseline_v1')
    parser.add_argument('--cache-path',default=None)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):audit_run(args.run_dir,args.cache_path)
    build_report(ROOT/'results',ROOT/'report')
    print('LaTeX inputs updated. Run python run.py bundle, then share results_to_share.zip.')


if __name__=='__main__':main()
