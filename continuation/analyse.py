"""Tables and figures from predictions; no fitting or hyperparameter selection."""
import argparse,json,itertools,csv
from pathlib import Path
import numpy as np
from common import save,metrics,vote,decomposition,paired_ci

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--bank',required=True);p.add_argument('--out',required=True);p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];preds={};y=np.load(Path(a.bank)/f'{a.split}_y.npy')
 for f in sorted(Path(a.results).glob('*/metrics.json')):
  r=json.loads(f.read_text());j=r['job'];d=f.parent
  filename=d/('predictions.npz' if a.split=='val' else 'test_predictions.npz')
  if not filename.exists():continue
  pp=np.load(filename)[a.split];preds[j['id']]=pp;mm=metrics(pp,y)
  rows.append(dict(**j,accuracy=mm['accuracy'],error=1-mm['accuracy'],log_loss=mm['log_loss'],brier=mm['brier'],train_error=r['train_bank']['errors']/r['train_bank']['n'],original_train_error=r['train_original_fit_labels']['errors']/r['train_original_fit_labels']['n'],probe_error=1-r.get('unseen_views_fit_labels',{'accuracy':0})['accuracy'],all_views_correct=r['all_views_correct_fraction'],n_original=r['n_original'],head_parameters=r['head_parameters'],trunk_parameters=r['trunk_parameters'],nodes=r['nodes'],seconds=r['training_seconds']))
 if not rows:raise ValueError('No completed predictions')
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with open(out/'all_runs.csv','w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for depth,noise in itertools.product([0,2],[0.,.15]):
  rr=[r for r in rows if r['phase']=='capacity' and r['depth']==depth and r['noise']==noise]
  if not rr:continue
  fig,ax=plt.subplots(1,3,figsize=(13,3.7))
  for mode in sorted({r['mode'] for r in rr}):
   ww=sorted({r['width'] for r in rr if r['mode']==mode})
   for axis,key,title in zip(ax,['train_error','error','log_loss'],['Fitted-bank train error',a.split+' classification error',a.split+' log loss']):
    vals=[[r[key] for r in rr if r['mode']==mode and r['width']==w] for w in ww]
    axis.errorbar(ww,[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in vals],label=mode,marker='o',capsize=3)
    axis.set_xscale('log',base=2);axis.set_xlabel('Hidden width');axis.set_title(title);axis.grid(alpha=.2)
   ax[0].legend()
  fig.tight_layout();fig.savefig(out/f'capacity_depth{depth}_noise{noise}.png',dpi=160);plt.close(fig)
 # Complete crossed data-seed x initialization design; population-moment denominator R makes identity exact.
 bv=[]
 for mode,fraction,epochs in sorted({(r['mode'],r['fraction'],r['epochs']) for r in rows if r['phase']=='biasvariance'}):
  rr=[r for r in rows if r['phase']=='biasvariance' and (r['mode'],r['fraction'],r['epochs'])==(mode,fraction,epochs)]
  ds=sorted({r['data_seed'] for r in rr});ss=sorted({r['seed'] for r in rr})
  complete=len(rr)==len(ds)*len(ss) and len(ds)>=2 and len(ss)>=2
  if not complete:continue
  P=np.stack([[preds[next(r['id'] for r in rr if r['data_seed']==d and r['seed']==s)] for s in ss] for d in ds]).astype(float)
  dmean=P.mean(1);grand=P.mean((0,1));within=np.square(P-dmean[:,None]).sum(-1).mean();between=np.square(dmean-grand).sum(-1).mean()
  dec=decomposition(P.reshape(-1,*P.shape[2:]),y)
  assert np.allclose(within+between,dec['prediction_variance'],atol=1e-10)
  bv.append(dict(mode=mode,fraction=fraction,epochs=epochs,replicates=len(rr),**dec,within_dataset_seed_variance=float(within),between_dataset_mean_variance=float(between),note='Empirical conditional decomposition; between term includes finite-initialization averaging error. Not unbiased population components.'))
 save(out/'bias_variance.json',bv)
 if bv:
  fig,axs=plt.subplots(1,3,figsize=(13,3.6))
  for mode,fraction in sorted({(r['mode'],r['fraction']) for r in bv}):
   rr=sorted([r for r in bv if r['mode']==mode and r['fraction']==fraction],key=lambda r:r['epochs'])
   for ax,key in zip(axs,['bias_related_proxy','prediction_variance','mean_brier']):
    ax.plot([r['epochs'] for r in rr],[r[key] for r in rr],marker='o',label=f'{mode}, n={fraction:.0%}');ax.set_title(key);ax.set_xlabel('Per-node epoch budget');ax.grid(alpha=.2)
  axs[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'bias_variance.png',dpi=160);plt.close(fig)
 ensembles=[]
 for family in sorted({r['ensemble_family'] for r in rows if r['ensemble_family']}):
  rr=sorted([r for r in rows if r['ensemble_family']==family],key=lambda r:r['seed']);P=np.stack([preds[r['id']] for r in rr]);labels=P.argmax(-1);K=P.shape[-1]
  for M in [1,3,5,7]:
   if M>len(rr):continue
   acc=[];tie=[]
   for comb in itertools.combinations(range(len(rr)),M):
    v,t=vote(labels[list(comb)],K);acc.append(float((v==y).mean()));tie.append(t)
   vv,_=vote(labels[:M],K)
   pairs=list(itertools.combinations(range(M),2));dis=[float((labels[i]!=labels[j]).mean()) for i,j in pairs];double=[float(((labels[i]!=y)&(labels[j]!=y)).mean()) for i,j in pairs]
   ensembles.append(dict(family=family,M=M,accuracy_mean_all_subsets=float(np.mean(acc)),accuracy_min=float(min(acc)),accuracy_max=float(max(acc)),tie_fraction_mean=float(np.mean(tie)),canonical_accuracy=float((vv==y).mean()),canonical_member_ids=[r['id'] for r in rr[:M]],canonical_disagreement=float(np.mean(dis)) if dis else 0.,canonical_double_fault=float(np.mean(double)) if double else 0.,head_parameters=sum(r['head_parameters'] for r in rr[:M]),total_parameters=rr[0]['trunk_parameters']+sum(r['head_parameters'] for r in rr[:M]),paired_vs_first_member=paired_ci(labels[0],vv,y),note='Subset ranges are descriptive, not confidence intervals. Canonical prefix order fixed by numeric training seed; no best subset selected.'))
 save(out/'ensembles.json',ensembles)
 if ensembles:
  fig,ax=plt.subplots(figsize=(6,4))
  for fam in sorted({r['family'] for r in ensembles}):
   rr=[r for r in ensembles if r['family']==fam];ax.plot([r['M'] for r in rr],[100*r['accuracy_mean_all_subsets'] for r in rr],marker='o',label=fam)
  ax.set(xlabel='Number of complete predictors',ylabel=a.split+' accuracy (%)');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'ensemble_accuracy.png',dpi=160);plt.close(fig)
 # Actual saved root-only training trajectories, distinct from whole-tree budget fits.
 for f in sorted(Path(a.results).glob('*/metrics.json')):
  r=json.loads(f.read_text());tr=r.get('root_epoch_trajectory',[])
  if not tr:continue
  fig,ax=plt.subplots(1,2,figsize=(9,3))
  for split in ['train','val']:
   ax[0].plot([t['epoch'] for t in tr],[1-t[split]['accuracy'] for t in tr],label=split)
   ax[1].plot([t['epoch'] for t in tr],[t[split]['log_loss'] for t in tr],label=split)
  for x in ax:x.set_xlabel('Actual root-training epoch');x.legend();x.grid(alpha=.2)
  ax[0].set_ylabel('Classification error');ax[1].set_ylabel('Log loss');fig.tight_layout();fig.savefig(out/(r['job']['id']+'_trajectory.png'),dpi=160);plt.close(fig)
 save(out/'coverage.json',dict(completed=len(rows),phases={phase:sum(r['phase']==phase for r in rows) for phase in sorted({r['phase'] for r in rows})},biasvariance_cells=len(bv),ensemble_cells=len(ensembles),split=a.split,conclusions='No automatic claim of double descent, accuracy improvement or population bias. Inspect all curves, replicate spread and paired comparisons.'))
 print('Analysis saved:',out)
if __name__=='__main__':main()
