"""Resumable, manifest-driven fits; no test evaluation during development."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,time,pickle,copy,platform
from pathlib import Path
import numpy as np
from common import save,digest,source_hash,probabilities,metrics,nested_indices

def arrays(bank,job):
 b=Path(bank);y=np.load(b/'train_y.npy');idx=nested_indices(y,job['fraction'],job['data_seed'])
 original=np.load(b/'train_X.npy',mmap_mode='r');mean=original[idx].mean(0);std=np.maximum(original[idx].std(0),1e-5)
 # Fixed noise draw on original image IDs; each view inherits that same label.
 labels=y.copy();rng=np.random.default_rng(job['noise_seed']);flip=rng.permutation(len(y))[:int(len(y)*job['noise'])]
 K=len(np.unique(y));labels[flip]=(labels[flip]+rng.integers(1,K,len(flip)))%K
 mode=job['mode'];views=[original]
 if mode=='repeat':views += [original]*job['views']
 elif mode in ('augment','persistent'):
  for v in range(job['views']):views.append(np.load(b/f"view_{job['bank_id']*job['views']+v}.npy",mmap_mode='r'))
 X=np.concatenate([(v[idx]-mean)/std for v in views]).astype('float32');target=np.tile(labels[idx],len(views));groups=np.tile(idx,len(views))
 return X,target,groups,idx,mean,std,y,labels

def execute(bank,out,job):
 import torch
 from grouped_model import TrainConfig,Forest
 d=Path(out)/job['id'];d.mkdir(parents=True,exist_ok=True)
 request=dict(job=job,source=source_hash(),bank=digest(Path(bank)/'metadata.json'))
 if (d/'request.json').exists() and json.loads((d/'request.json').read_text())!=request:raise ValueError('Changed source/configuration for '+str(d))
 if (d/'metrics.json').exists():
  old=json.loads((d/'metrics.json').read_text())
  for name,key in [('model.pkl','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
   if not (d/name).exists() or digest(d/name)!=old.get(key):raise ValueError('Completed result missing or changed: '+str(d/name))
  print('DONE',job['id'],flush=True);return
 save(d/'request.json',request);X,y,groups,idx,mean,std,clean,noisy=arrays(bank,job)
 cfg=TrainConfig(width=job['width'],depth=job['depth'],epochs=job['epochs'],seed=job['seed'],device='cpu',fixed_epochs=True,log_every=50)
 cfg.persistent=job['mode']=='persistent';cfg.checkpoint_epochs=job.get('checkpoints',[]) if job['depth']==0 else []
 t=time.perf_counter();model=Forest(cfg,np.arange(len(np.unique(clean)))).fit(X,y,groups)
 seconds=time.perf_counter()-t;meta=json.loads((Path(bank)/'metadata.json').read_text())
 result=dict(job=job,training_seconds=seconds,n_original=len(idx),n_presented=len(y),nodes=len(model.nodes()),head_parameters=model.n_parameters(),trunk_parameters=meta['trunk_parameters'],train_bank=metrics(probabilities(model,X),y),node_stats=[dict(v.stats,history=v.history) for v in model.nodes()],environment=dict(platform=platform.platform(),torch=torch.__version__,numpy=np.__version__))
 result['optimizer_updates']=sum(int(np.ceil(v.stats['n']/cfg.batch_size))*v.history[-1]['epoch'] for v in model.nodes() if v.history)
 result['all_views_correct_fraction']=float((model.predict(X).reshape(-1,len(idx))==noisy[idx]).all(0).mean())
 pred={}
 for split in ('train','val'):
  xx=np.load(Path(bank)/f'{split}_X.npy',mmap_mode='r');yy=np.load(Path(bank)/f'{split}_y.npy')
  if split=='train':xx=xx[idx];yy=yy[idx]
  pp=probabilities(model,((xx-mean)/std).astype('float32'));pred[split]=pp.astype('float32');result[split+'_clean']=metrics(pp,yy)
  if split=='train':result['train_original_fit_labels']=metrics(pp,noisy[idx])
 probe=[]
 for f in sorted(Path(bank).glob('probe_*.npy')):
  xx=np.load(f,mmap_mode='r')[idx];probe.append(probabilities(model,((xx-mean)/std).astype('float32')))
 if probe:
  pp=np.stack(probe);result['unseen_views_clean']=metrics(pp.reshape(-1,pp.shape[-1]),np.tile(clean[idx],len(probe)))
  result['unseen_views_fit_labels']=metrics(pp.reshape(-1,pp.shape[-1]),np.tile(noisy[idx],len(probe)))
  result['probe_disagreement_with_original']=float((pp.argmax(-1)!=pred['train'].argmax(-1)).mean())
 # True epoch trajectories only for the root-only control. Full trees use independent budget fits.
 trajectory=[]
 for e in cfg.checkpoint_epochs:
  snapshot=copy.deepcopy(model)
  if not all(e in r.snapshots for r in snapshot.roots):continue
  for r in snapshot.roots:r.model.load_state_dict(r.snapshots[e])
  xv=(np.load(Path(bank)/'val_X.npy')-mean)/std;yv=np.load(Path(bank)/'val_y.npy')
  trajectory.append(dict(epoch=e,train=metrics(probabilities(snapshot,X),y),val=metrics(probabilities(snapshot,xv),yv)))
 result['root_epoch_trajectory']=trajectory
 for node in model.nodes():node.snapshots={}
 payload=dict(model=model,mean=mean,std=std,indices=idx,job=job,bank_request=meta['request'])
 with open(d/'model.tmp','wb') as f:pickle.dump(payload,f)
 (d/'model.tmp').replace(d/'model.pkl')
 np.savez_compressed(d/'predictions.npz',**pred,train_ids=idx)
 result['checkpoint_sha256']=digest(d/'model.pkl');result['predictions_sha256']=digest(d/'predictions.npz')
 save(d/'metrics.json',result)
 print('SAVED',job['id'],'seconds',round(seconds,2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
 p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--threads',type=int,default=4);p.add_argument('--limit',type=int,default=0)
 a=p.parse_args();import torch
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(a.threads);jobs=json.loads(Path(a.plan).read_text())['jobs'];bank=Path(a.bank)
 meta=json.loads((bank/'metadata.json').read_text())
 for name,h in meta['hashes'].items():
  if digest(bank/name)!=h:raise ValueError('Bank hash mismatch: '+name)
 selected=[j for i,j in enumerate(jobs) if i%a.shards==a.shard]
 if a.limit:selected=selected[:a.limit]
 if (Path(a.out)/'TEST_LOCK.json').exists():raise ValueError('Test evaluated: use a new output directory; do not retune against test scores')
 with threadpool_limits(limits=a.threads):
  for j in selected:execute(bank,a.out,j)
if __name__=='__main__':main()
