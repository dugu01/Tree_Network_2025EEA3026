#!/usr/bin/env python3
"""Finish the prespecified comparisons using existing checkpoints only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import time
from run import (read_model,save_model,write_metrics,source_digest,finalize,bundle)
from tree_model import prune_forest
from data_features import load_features,save_json,digest_file
from extras.review_controls import audit_run
from explain import explain_run
from reporting import build_report
import torch
from threadpoolctl import threadpool_limits

SOURCE='f9efc1a849247bef675e6d6cbb653176fcf231a182fe56ab15b76d4469167fd9'

def selected_runs():
    rows=[]
    for dataset,width in [('cifar10',16),('catsdogs',8),('catsdogs',16)]:
        for seed in [17,29,43]:
            rows.append((dataset,width,seed,'branched comparison'))
    rows.extend([('cifar10',64,17,'smaller root-only comparator'),
                 ('cifar10',256,17,'highest mean validation accuracy in CIFAR sweep'),
                 ('catsdogs',32,17,'root-only comparator for seed 17')])
    return rows

def preflight(results):
    if source_digest()!=SOURCE:
        raise ValueError('Core source differs from the reviewed sweep version. Do not overwrite results; share this message.')
    entries=[]
    for dataset,width,seed,role in selected_runs():
        run=results/dataset/'main'/f'sweep_capacity_clean_v1_w{width}_s{seed}'
        required=[run/'request.json',run/'cache_reference.json']
        required += [run/kind/name for kind in ['root','tree'] for name in ['model.pkl','metrics.json']]
        missing=[str(p) for p in required if not p.is_file()]
        if missing:raise FileNotFoundError('Missing completed sweep files: '+', '.join(missing))
        request=json.loads((run/'request.json').read_text());cfg=request['config']
        if (request['source_sha256']!=SOURCE or cfg['width']!=width or cfg['seed']!=seed
                or cfg['epochs']!=100 or cfg['depth']!=2 or not cfg['fixed_epochs']):
            raise ValueError(f'Unexpected experiment configuration: {run}')
        cache=Path(json.loads((run/'cache_reference.json').read_text())['path'])
        meta=json.loads((cache/'metadata.json').read_text())
        if meta['manifest_sha256']!=request['cache_manifest_sha256']:
            raise ValueError(f'Cache/run mismatch: {run}')
        if digest_file(cache/'split_manifest.json')!=meta['manifest_sha256']:
            raise ValueError(f'Modified manifest: {cache}')
        entries.append({'dataset':dataset,'width':width,'seed':seed,'role':role,
                        'run':str(run.relative_to(results)),
                        'root_checkpoint_sha256':digest_file(run/'root'/'model.pkl'),
                        'tree_checkpoint_sha256':digest_file(run/'tree'/'model.pkl'),
                        'cache_manifest_sha256':request['cache_manifest_sha256']})
    # Hash each distinct cache once. No training is performed by this helper.
    caches={Path(json.loads((results/e['run']/'cache_reference.json').read_text())['path']) for e in entries}
    for cache in caches:
        meta=json.loads((cache/'metadata.json').read_text())
        for name,expected in meta['file_hashes'].items():
            if digest_file(cache/name)!=expected:raise ValueError(f'Modified feature cache: {cache/name}')
    return entries

def finish(results,device,dry_run=False):
    entries=preflight(results)
    print(f'Preflight passed: {len(entries)} existing runs; no training required.',flush=True)
    if dry_run:return
    protocol={'version':1,'basis':'Three-seed training/validation sweep; no test scores used to choose this set.',
              'interpretation':'Tree/pruned interpolation comparisons are reported alongside root controls; do not select models retrospectively using test scores.',
              'pruning':'Greedy subtree removal: zero training-accuracy drop and at most 0.5 percentage-point validation drop from the unpruned model.',
              'within_run_selection':'Existing finalize rule: highest validation accuracy, then fewer total parameters; all four variants receive test evaluation.',
              'seed_policy':'17,29,43 for the three branched configurations; seed 17 fixed for the additional root-only comparators and visual examples.',
              'core_source_sha256':SOURCE,'helper_sha256':digest_file(__file__),'runs':entries}
    protocol_path=results/'final_evaluation_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:
            raise ValueError('A different final protocol already exists. Keep its evaluation fixed.')
    else:
        if any(json.loads(p.read_text()).get('test') is not None
               for e in entries for p in (results/e['run']).glob('*/metrics.json')):
            raise ValueError('Test scores already exist without this protocol. Share the results before proceeding.')
        save_json(protocol_path,protocol)
    # Finish all pruning/controls/visuals BEFORE computing any test accuracy.
    for i,e in enumerate(entries,1):
        run=results/e['run'];dest=run/'pruned'
        print(f'[{i}/{len(entries)}] Preparing {e["run"]}',flush=True)
        if (run/'selection_locked.json').exists():
            if not (dest/'metrics.json').exists() or not (run/'control_audit.json').exists():
                raise ValueError(f'Finalized run is missing preparation outputs: {run}')
            continue
        if not (dest/'metrics.json').exists():
            cache=Path(json.loads((run/'cache_reference.json').read_text())['path'])
            arrays,meta,mean,std=load_features(cache)
            checkpoint=read_model(run/'tree'/'model.pkl');start=time.perf_counter()
            pruned,events=prune_forest(checkpoint['model'],*arrays['train'],*arrays['val'],
                                       max_train_drop=0.,max_val_drop=.005)
            dest.mkdir(exist_ok=True)
            save_model(dest/'model.pkl',pruned,mean,std,meta)
            cfg=json.loads((run/'request.json').read_text())['config']
            write_metrics(dest,pruned,arrays,meta,cfg,'pruned',start,events)
            del arrays,pruned,checkpoint
        audit_run(run)
        if e['seed']==17 and e['role']=='branched comparison':
            explain_run(run,device=device,data_root=str(ROOT/'data'),max_nodes=64)
    print('Preparation complete. Evaluating the prespecified test comparisons.',flush=True)
    for e in entries:finalize(results/e['run'])
    build_report(results,ROOT/'report',compile_pdf=False)
    bundle(results,ROOT/'results_to_share.zip')
    print('DONE. Share results_to_share.zip. LaTeX inputs updated; final prose/PDF review remains.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=4):finish(ROOT/'results',args.device,args.dry_run)

if __name__=='__main__':main()
