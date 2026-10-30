"""Pure numerical analysis; test labels never select a model."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
from scipy.special import expit,softmax

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

def source_hash():
 p=Path(__file__).parent;h=hashlib.sha256()
 for f in sorted(p.glob('*.py')):h.update(f.name.encode());h.update(f.read_bytes())
 return h.hexdigest()

def probabilities(model,X):
 s=model.scores(X)
 return np.column_stack((1-expit(s[:,0]),expit(s[:,0]))) if len(model.classes)==2 else softmax(s,axis=1)

def metrics(p,y):
 p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int)
 pred=p.argmax(-1);one=np.eye(p.shape[-1])[y]
 return dict(n=len(y),errors=int((pred!=y).sum()),accuracy=float((pred==y).mean()),
  log_loss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
  brier=float(np.square(p-one).sum(-1).mean()),confusion=np.bincount(y*p.shape[-1]+pred,minlength=p.shape[-1]**2).reshape(p.shape[-1],-1).tolist())

def vote(pred,K):
 """Plurality/majority voting; ties resolved by smallest class index."""
 pred=np.asarray(pred,dtype=int)
 counts=(pred[...,None]==np.arange(K)).sum(0)
 ties=(counts==counts.max(-1,keepdims=True)).sum(-1)>1
 return counts.argmax(-1),float(ties.mean())

def decomposition(p,y):
 p=np.asarray(p,dtype=float);mean=p.mean(0);target=np.eye(p.shape[-1])[y]
 bias_proxy=np.square(mean-target).sum(-1).mean()
 variance=np.square(p-mean).sum(-1).mean()
 risk=np.square(p-target).sum(-1).mean()
 assert np.allclose(risk,bias_proxy+variance,atol=1e-10)
 return dict(bias_related_proxy=float(bias_proxy),prediction_variance=float(variance),mean_brier=float(risk),residual=float(risk-bias_proxy-variance))

def paired_ci(a,b,y,seed=2026,draws=1000):
 """Paired image bootstrap for accuracy difference b-a; no training-set uncertainty."""
 delta=(np.asarray(b)==y).astype(float)-(np.asarray(a)==y).astype(float)
 rng=np.random.default_rng(seed);means=[delta[rng.integers(len(y),size=len(y))].mean() for _ in range(draws)]
 lo,hi=np.quantile(means,[.025,.975])
 return dict(difference=float(delta.mean()),low=float(lo),high=float(hi),unit='fraction',scope='paired evaluation images, conditional on trained models')

def nested_indices(y,fraction,seed):
 """Nested stratified subsets for each independent data seed."""
 rng=np.random.default_rng(seed);idx=[]
 for c in np.unique(y):
  ids=rng.permutation(np.flatnonzero(y==c));idx.extend(ids[:max(1,int(len(ids)*fraction))])
 return np.sort(idx)
