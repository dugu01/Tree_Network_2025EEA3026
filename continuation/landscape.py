"""Fixed-data, row-normalized slices of the actual hard-correction model."""
import argparse,pickle,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import save

def loss(model,x,y):
 scores=model.scores(x)
 target=(y[:,None]==model.classes[None,:]).astype(float) if len(model.classes)>2 else y[:,None]
 return float((np.logaddexp(0,scores)-target*scores).mean())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--bank',required=True);p.add_argument('--out',required=True);p.add_argument('--grid',type=int,default=15);p.add_argument('--radius',type=float,default=.3);p.add_argument('--samples',type=int,default=512);p.add_argument('--directions',type=int,default=3);a=p.parse_args()
 import torch
 with open(Path(a.run)/'model.pkl','rb') as f:payload=pickle.load(f)
 model=payload['model'];bank=Path(a.bank);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 fixed={};rng=np.random.default_rng(901)
 for split in ['train','val']:
  
  if split=='train':
   from experiment import arrays
   x,y,*_=arrays(bank,payload['job'])
  else:
   x=((np.load(bank/'val_X.npy')-payload['mean'])/payload['std']).astype('float32');y=np.load(bank/'val_y.npy')
  ids=rng.choice(len(y),size=min(a.samples,len(y)),replace=False)
  fixed[split]=(x[ids],y[ids]);save(out/(split+'_indices.json'),ids.tolist())
 params=[p for node in model.nodes() if node.model is not None for p in node.model.parameters()];original=[p.detach().clone() for p in params]
 # Biases and nonnegative correction scalars stay fixed. MLP row norms normalize each direction.
 def direction(seed):
  rng=np.random.default_rng(seed);d=[]
  for v in original:
   vv=v.numpy();z=rng.normal(size=vv.shape)
   if vv.ndim<2:z*=0
   else:z*=np.linalg.norm(vv,axis=1,keepdims=True)/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
   d.append(torch.tensor(z,dtype=v.dtype))
  return d
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 grid=np.linspace(-a.radius,a.radius,a.grid);base={s:loss(model,*xy) for s,xy in fixed.items()};summary=[]
 try:
  for seed in range(a.directions):
   d1,d2=direction(100+seed*2),direction(101+seed*2);surfaces={s:np.zeros((a.grid,a.grid)) for s in fixed}
   flip=np.zeros((a.grid,a.grid));base_preds=model.predict(fixed['val'][0])
   for i,alpha in enumerate(grid):
    for j,beta in enumerate(grid):
     with torch.no_grad():
      for p,v,u,w in zip(params,original,d1,d2):p.copy_(v+alpha*u+beta*w)
     for split,xy in fixed.items():surfaces[split][j,i]=loss(model,*xy)
     flip[j,i]=np.mean(model.predict(fixed['val'][0])!=base_preds)
   with torch.no_grad():
    for p,v in zip(params,original):p.copy_(v)
   np.savez_compressed(out/f'slice_{seed}.npz',axis=grid,decision_flip=flip,**surfaces)
   fig,axs=plt.subplots(1,3,figsize=(12,3.4))
   for ax,(name,z) in zip(axs,list(surfaces.items())+[('decision flip fraction',flip)]):
    im=ax.contourf(grid,grid,z,levels=20);fig.colorbar(im,ax=ax);ax.set(title=name,xlabel='alpha',ylabel='beta')
   fig.tight_layout();fig.savefig(out/f'slice_{seed}.png',dpi=160);plt.close(fig)
   summary.append(dict(direction_seed=seed,max_train_increase=float(surfaces['train'].max()-base['train']),max_val_increase=float(surfaces['val'].max()-base['val'])))
 finally:
  with torch.no_grad():
   for p,v in zip(params,original):p.copy_(v)
 # Exact PSD Hessian for conditional ROOT readout, on fixed training samples only.
 hess=[];x,y=fixed['train']
 for root in model.roots:
  if root.model is None:continue
  H=root.hidden(x).astype(float);pa=root.A.predict(x) if root.A else np.zeros(len(x));pb=root.B.predict(x) if root.B else np.zeros(len(x))
  D=np.column_stack([H,np.ones(len(x)),pa,-pb]);s=root.score(x);q=expit(s);reg=np.ones(D.shape[1]);reg[H.shape[1]]=0
  matrix=(D.T*(q*(1-q)))@D/len(x)+root.config.refit_l2*np.diag(reg);ev=np.linalg.eigvalsh(matrix)
  hess.append(dict(node=root.name,smallest=float(ev[0]),largest=float(ev[-1]),psd=bool(ev[0]>=-1e-8),samples=len(x),scope='conditional unweighted regularized readout, children and hidden features held fixed; subsample diagnostic not full-data optimizer Hessian'))
 save(out/'landscape.json',dict(base_loss=base,slices=summary,conditional_readout=hess,objective='Unweighted mean binary cross entropy across OVR root outputs; distinct from class-balanced local node training loss and multiclass softmax test log loss',interpretation='Hard child decisions recomputed at every perturbation. A slice is not a global convexity/flatness proof; biases and edge scalars fixed. No global full-tree Hessian is claimed.'))
 print('Landscape saved:',out)
if __name__=='__main__':main()
