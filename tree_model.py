"""Neural positive/negative correction trees following Jayadeva's Table II.

Training uses cached features; only the output-layer refit is convex.
No accuracy or double-descent outcome is assumed.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class TrainConfig:
    width: int = 64
    depth: int = 2
    epochs: int = 80
    batch_size: int = 512
    lr: float = 0.001
    seed: int = 17
    min_samples: int = 16
    device: str = "cpu"
    refit_l2: float = 1e-6
    refit_iterations: int = 120
    fixed_epochs: bool = False
    log_every: int = 10


def categories(score, y):
    pred = np.asarray(score) >= 0
    y = np.asarray(y).astype(bool)
    return ~pred & ~y, pred & y, ~pred & y, pred & ~y


def bce(score, y):
    return float(np.mean(np.logaddexp(0, score) - y * score))


def make_mlp(d, h):
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))


class Node:
    def __init__(self, d, config, name="root", depth=0, seed=None):
        self.config = config
        self.name, self.depth = name, depth
        self.seed = config.seed if seed is None else seed
        self.d = d
        self.model = None
        self.constant = None
        self.A = self.B = None
        self.a = self.b = 0.0
        self.stats = {}
        self.history = []

    def base_score(self, X):
        if self.constant is not None:
            return np.full(len(X), 1.0 if self.constant else -1.0)
        # NumPy avoids accelerator launch overhead for small inference calls.
        w1, c1, w2, c2 = [p.detach().cpu().numpy() for p in self.model.parameters()]
        return (np.maximum(X @ w1.T + c1, 0) @ w2.T + c2).ravel().astype(np.float64)

    def hidden(self, X):
        w1 = self.model[0].weight.detach().cpu().numpy()
        c1 = self.model[0].bias.detach().cpu().numpy()
        return np.maximum(X @ w1.T + c1, 0)

    def score(self, X):
        s = self.base_score(X)
        if self.A is not None:
            s += self.a * self.A.predict(X)
        if self.B is not None:
            s -= self.b * self.B.predict(X)
        return s

    def predict(self, X):
        return (self.score(X) >= 0).astype(np.int64)

    def predict_certified(self, X, tolerance=1e-7):
        """Exact binary decision with conservative interval-based skipping.

        Returns decisions and the number of node-example evaluations.
        These are operation counts, not a wall-clock speed claim.
        """
        s = self.base_score(X)
        n = len(X)
        lo, hi = s - self.b, s + self.a
        positive = lo > tolerance
        negative = hi < -tolerance
        uncertain = ~(positive | negative)
        out = positive.astype(np.int64)
        evaluations = n
        if uncertain.any():
            z, t = X[uncertain], s[uncertain].copy()
            if self.A is not None:
                p, count = self.A.predict_certified(z, tolerance)
                t += self.a * p
                evaluations += count
            if self.B is not None:
                p, count = self.B.predict_certified(z, tolerance)
                t -= self.b * p
                evaluations += count
            out[uncertain] = (t >= 0).astype(np.int64)
        return out, evaluations

    def walk(self):
        yield self
        if self.A is not None:
            yield from self.A.walk()
        if self.B is not None:
            yield from self.B.walk()

    def n_parameters(self):
        return sum((sum(p.numel() for p in v.model.parameters()) if v.model is not None else 1)
                   + int(v.A is not None) + int(v.B is not None) for v in self.walk())

    def fit(self, X, y):
        c = self.config
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.stats = {"name": self.name, "depth": self.depth, "n": len(y),
                      "positives": int(y.sum()), "seed": self.seed}
        start = time.perf_counter()
        if len(np.unique(y)) == 1:
            self.constant = int(y[0])
            self.stats.update(base_errors=0, final_errors=0, seconds=0., refit="constant")
            return self
        torch.manual_seed(self.seed)
        self.model = make_mlp(X.shape[1], c.width).to(c.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=c.lr, weight_decay=0.)
        xt = torch.from_numpy(X).to(c.device)
        yt = torch.from_numpy(y.astype(np.float32)).to(c.device)
        # Normalize class-balanced weights to mean one.
        n1 = int(y.sum()); n0 = len(y) - n1
        wt = torch.where(yt > 0, len(y)/(2*n1), len(y)/(2*n0))
        rng = np.random.default_rng(self.seed)
        for epoch in range(1, c.epochs + 1):
            self.model.train()
            perm = rng.permutation(len(y))
            for j in range(0, len(y), c.batch_size):
                idx = torch.as_tensor(perm[j:j+c.batch_size], device=c.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xt[idx]).ravel()
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, yt[idx], reduction="none") * wt[idx]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {self.name}")
                loss.backward(); optimizer.step()
            if epoch == 1 or epoch % c.log_every == 0 or epoch == c.epochs:
                self.model.eval()
                with torch.no_grad():
                    pred = torch.cat([self.model(xt[j:j+c.batch_size]).ravel()
                                      for j in range(0, len(y), c.batch_size)])
                    errors = int(((pred >= 0) != (yt > 0)).sum().item())
                    raw_loss = float(nn.functional.binary_cross_entropy_with_logits(pred, yt).item())
                self.history.append({"epoch": epoch, "errors": errors, "loss": raw_loss})
                print(f"  {self.name}: epoch {epoch}/{c.epochs}, errors {errors}/{len(y)}", flush=True)
                if errors == 0 and not c.fixed_epochs and epoch >= 10:
                    break
        self.model.cpu().eval()
        del optimizer, xt, yt, wt
        if c.device == "mps":
            torch.mps.empty_cache()
        s = self.base_score(X)
        C1,C2,C3,C4 = categories(s, y)
        base_errors = int((C3 | C4).sum())
        self.stats["base_errors"] = base_errors
        if self.depth < c.depth and base_errors and len(y) >= c.min_samples:
            if C3.any():
                mask = ~C2
                self.A = Node(self.d,c,self.name+".A",self.depth+1,self.seed*2+1).fit(X[mask],C3[mask].astype(int))
            if C4.any():
                mask = ~C1
                self.B = Node(self.d,c,self.name+".B",self.depth+1,self.seed*2+2).fit(X[mask],C4[mask].astype(int))
            self._refit(X,y,s)
        final_errors = int(np.sum(self.predict(X) != y))
        if final_errors > base_errors:
            raise AssertionError("Accepted correction worsened local training error")
        self.stats.update(final_errors=final_errors,seconds=time.perf_counter()-start,
                          a=float(self.a),b=float(self.b))
        return self

    def _refit(self, X, y, original_score):
        """Convex logistic output refit, then explicit empirical acceptance.

        Convexity does not imply monotonic 0/1 accuracy. Compare with the
        unchanged parent; discard children when the candidate is worse.
        """
        c = self.config
        H = self.hidden(X).astype(np.float64)
        pa = self.A.predict(X) if self.A is not None else np.zeros(len(y))
        pb = self.B.predict(X) if self.B is not None else np.zeros(len(y))
        design = np.column_stack([H,np.ones(len(y)),pa,-pb])
        h = H.shape[1]
        initial = np.r_[self.model[2].weight.detach().numpy().ravel(),
                        self.model[2].bias.detach().numpy(),0.,0.].astype(np.float64)
        reg = np.ones(h+3);reg[h]=0
        def fun(w):
            score=design@w
            value=np.mean(np.logaddexp(0,score)-y*score)+.5*c.refit_l2*np.sum(reg*w*w)
            grad=design.T@(expit(score)-y)/len(y)+c.refit_l2*reg*w
            return float(value),grad
        result=minimize(fun,initial,jac=True,method="L-BFGS-B",
                        bounds=[(None,None)]*(h+1)+[(0,None if self.A is not None else 0),
                                                (0,None if self.B is not None else 0)],
                        options={"maxiter":c.refit_iterations,"ftol":1e-10})
        candidate=result.x
        score=design@candidate
        old_error=int(np.sum((original_score>=0)!=y))
        new_error=int(np.sum((score>=0)!=y))
        accept=np.isfinite(score).all() and (new_error,bce(score,y)) < (old_error,bce(original_score,y))
        self.stats.update(refit_success=bool(result.success),refit_message=str(result.message),
                          refit_candidate_errors=new_error,refit_accepted=bool(accept))
        if accept:
            old_w=self.model[2].weight.detach().clone();old_b=self.model[2].bias.detach().clone()
            with torch.no_grad():
                self.model[2].weight.copy_(torch.tensor(candidate[:h],dtype=torch.float32)[None,:])
                self.model[2].bias.copy_(torch.tensor(candidate[h:h+1],dtype=torch.float32))
            self.a,self.b=float(candidate[-2]),float(candidate[-1])
            # Recheck after conversion to the actual inference precision.
            if np.sum(self.predict(X)!=y)>old_error:
                with torch.no_grad():
                    self.model[2].weight.copy_(old_w);self.model[2].bias.copy_(old_b)
                accept=False
        if not accept:
            self.A=self.B=None;self.a=self.b=0.
            self.stats["refit_accepted"]=False
        else:
            if self.a==0:self.A=None
            if self.b==0:self.B=None
        final=self.predict(X)
        self.stats.update(fixed=int(np.sum(((original_score>=0)!=y)&(final==y))),
                          spoiled=int(np.sum(((original_score>=0)==y)&(final!=y))))


class Forest:
    def __init__(self, config, classes):
        self.config=config
        self.classes=np.asarray(classes)
        self.roots=[]

    def fit(self,X,y):
        targets=[self.classes[1]] if len(self.classes)==2 else list(self.classes)
        self.roots=[Node(X.shape[1],self.config,f"class_{int(k)}",seed=self.config.seed+1009*j).fit(X,(y==k).astype(int))
                    for j,k in enumerate(targets)]
        return self

    def scores(self,X):
        return np.column_stack([r.score(X) for r in self.roots])

    def predict(self,X):
        s=self.scores(X)
        if len(self.classes)==2:return self.classes[(s[:,0]>=0).astype(int)]
        return self.classes[np.argmax(s,axis=1)]

    def n_parameters(self):return sum(r.n_parameters() for r in self.roots)
    def nodes(self):return [v for r in self.roots for v in r.walk()]

    def predict_certified(self,X):
        if len(self.classes)==2:
            p,c=self.roots[0].predict_certified(X)
            return self.classes[p],c
        # Certify root-score argmax only when a lower bound exceeds all other
        # upper bounds. Remaining examples use the complete original scores.
        base=np.column_stack([r.base_score(X) for r in self.roots])
        lo=base-np.array([r.b for r in self.roots])
        hi=base+np.array([r.a for r in self.roots])
        winner=np.argmax(lo,axis=1)
        competitors=hi.copy();competitors[np.arange(len(X)),winner]=-np.inf
        cert=lo[np.arange(len(X)),winner]>competitors.max(axis=1)+1e-7
        out=self.classes[winner].copy()
        evaluations=len(X)*len(self.roots)
        if (~cert).any():
            z=X[~cert];s=base[~cert].copy()
            for k,r in enumerate(self.roots):
                if r.A is not None:
                    p,count=r.A.predict_certified(z);s[:,k]+=r.a*p;evaluations+=count
                if r.B is not None:
                    p,count=r.B.predict_certified(z);s[:,k]-=r.b*p;evaluations+=count
            out[~cert]=self.classes[np.argmax(s,axis=1)]
        return out,evaluations


def prune_forest(model,Xtr,ytr,Xval,yval,max_train_drop=0.,max_val_drop=.005):
    """Greedy subtree pruning against fixed pre-pruning accuracy constraints."""
    out=copy.deepcopy(model)
    reference_train=np.mean(out.predict(Xtr)==ytr)
    reference_val=np.mean(out.predict(Xval)==yval)
    events=[]
    for parent in sorted(out.nodes(),key=lambda v:v.depth,reverse=True):
        for attr,weight in [("A","a"),("B","b")]:
            child=getattr(parent,attr)
            if child is None:continue
            w=getattr(parent,weight);setattr(parent,attr,None);setattr(parent,weight,0.)
            tr=np.mean(out.predict(Xtr)==ytr);va=np.mean(out.predict(Xval)==yval)
            keep=tr>=reference_train-max_train_drop-1e-12 and va>=reference_val-max_val_drop-1e-12
            events.append({"parent":parent.name,"child":attr,"removed":bool(keep),"train_accuracy":float(tr),"val_accuracy":float(va)})
            if not keep:setattr(parent,attr,child);setattr(parent,weight,w)
    return out,events
