"""Synthetic correctness tests; NOT CIFAR/CatsDogs scientific results."""
import unittest,tempfile
from pathlib import Path
import numpy as np
from common import vote,decomposition,nested_indices,save,digest
class Checks(unittest.TestCase):
 def test_multiclass_tie(self):
  pred,tie=vote(np.array([[2,1],[1,1],[0,2]]),3);np.testing.assert_array_equal(pred,[0,1]);self.assertEqual(tie,.5)
 def test_decomposition(self):
  rng=np.random.default_rng(2);p=rng.dirichlet([1,2,3],size=(5,31));d=decomposition(p,rng.integers(3,size=31));self.assertLess(abs(d['residual']),1e-12)
 def test_nested_samples(self):
  y=np.tile(np.arange(3),40);a=nested_indices(y,.1,17);b=nested_indices(y,.5,17);self.assertTrue(set(a)<=set(b));self.assertEqual(len(a),12)
 def test_views_and_noise(self):
  from experiment import arrays
  with tempfile.TemporaryDirectory() as d:
   make_bank(Path(d));j=job();j.update(noise=.25,mode='augment');X,y,groups,idx,mu,sd,clean,noisy=arrays(d,j);self.assertEqual(len(y),len(idx)*4)
   for block in y.reshape(4,-1):np.testing.assert_array_equal(block,noisy[idx])
   np.testing.assert_array_equal(groups.reshape(4,-1)[0],idx);self.assertEqual(np.sum(clean!=noisy),int(len(clean)*.25))
 def test_unmodified_equivalence(self):
  import torch
  from baseline.tree_model import TrainConfig as BC,Forest as BF
  from grouped_model import TrainConfig,Forest
  torch.set_num_threads(1);rng=np.random.default_rng(4);x=rng.normal(size=(40,6)).astype('float32');y=rng.integers(2,size=40)
  a=BF(BC(width=4,depth=1,epochs=4,seed=17,fixed_epochs=True),np.arange(2)).fit(x,y)
  b=Forest(TrainConfig(width=4,depth=1,epochs=4,seed=17,fixed_epochs=True),np.arange(2)).fit(x,y,np.repeat(np.arange(20),2))
  np.testing.assert_allclose(a.scores(x),b.scores(x),atol=0,rtol=0)
  for root in b.roots:self.assertLessEqual(root.stats['final_errors'],root.stats['base_errors'])
 def test_holdout_does_not_fit_normalizer(self):
  from experiment import arrays
  with tempfile.TemporaryDirectory() as d:
   make_bank(Path(d));j=job();first=arrays(d,j)
   np.save(Path(d)/'val_X.npy',np.ones((20,6))*99999);np.save(Path(d)/'test_y.npy',np.zeros(20))
   second=arrays(d,j)
   for i in [0,1,4,5]:np.testing.assert_array_equal(first[i],second[i])
 def test_corrupt_completed_result_refused(self):
  import torch
  from experiment import execute
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);make_bank(p/'bank');j=job();execute(p/'bank',p/'results',j)
   f=p/'results'/j['id']/'predictions.npz';f.write_bytes(b'changed')
   with self.assertRaises(ValueError):execute(p/'bank',p/'results',j)
 def test_multiclass_complete_predictor(self):
  import torch
  from grouped_model import TrainConfig,Forest
  from common import probabilities
  torch.set_num_threads(1);rng=np.random.default_rng(7);x=rng.normal(size=(36,6)).astype('float32');y=np.tile(np.arange(3),12)
  m=Forest(TrainConfig(width=4,depth=1,epochs=3,seed=17,fixed_epochs=True),np.arange(3)).fit(x,y)
  pp=probabilities(m,x);self.assertEqual(pp.shape,(36,3));np.testing.assert_allclose(pp.sum(1),1.)
  np.testing.assert_array_equal(pp.argmax(1),m.predict(x))
 def test_view_specific_targets(self):
  from grouped_model import categories
  c1,c2,c3,c4=categories(np.array([-1,1]),np.array([1,1]));np.testing.assert_array_equal(c3,[True,False]);np.testing.assert_array_equal(c2,[False,True])
def job():
 return dict(id='synthetic_0000',dataset='synthetic',phase='pilot',width=8,depth=1,epochs=5,fraction=1.,data_seed=101,seed=17,noise=0.,noise_seed=1234,mode='augment',views=3,bank_id=0,ensemble_family=None)
def make_bank(p):
 p.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(51)
 for split,n in [('train',48),('val',20),('test',20)]:
  x=rng.normal(size=(n,6)).astype('float32');y=(x[:,0]+x[:,1]>.0).astype(int);np.save(p/f'{split}_X.npy',x);np.save(p/f'{split}_y.npy',y)
 x=np.load(p/'train_X.npy')
 for kind,count in [('view',9),('probe',2)]:
  for v in range(count):np.save(p/f'{kind}_{v}.npy',(x+.03*rng.normal(size=x.shape)).astype('float32'))
 save(p/'metadata.json',dict(request={'dataset':'synthetic','note':'correctness only'},classes=['a','b'],counts={'train':48,'val':20,'test':20},trunk_parameters=0,hashes={f.name:digest(f) for f in p.glob('*.npy')}))
if __name__=='__main__':unittest.main(verbosity=2)
