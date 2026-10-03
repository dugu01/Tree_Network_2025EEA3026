import unittest
import numpy as np
import torch
from sklearn.datasets import make_moons
from tree_model import Node,Forest,TrainConfig,categories,make_mlp,prune_forest

torch.set_num_threads(2)


def random_node(d=4,depth=0):
    c=TrainConfig(width=5,depth=2)
    n=Node(d,c,depth=depth);n.model=make_mlp(d,5).eval()
    if depth<2:
        n.A=random_node(d,depth+1);n.B=random_node(d,depth+1);n.a=.8;n.b=.6
    return n


class TreeTests(unittest.TestCase):
    def test_category_targets(self):
        C1,C2,C3,C4=categories(np.array([-1,1,-1,1]),np.array([0,1,1,0]))
        np.testing.assert_array_equal(np.stack([C1,C2,C3,C4]),np.eye(4,dtype=bool))
        np.testing.assert_array_equal(C3[~C2],[0,1,0])
        np.testing.assert_array_equal(C4[~C1],[0,0,1])

    def test_signed_corrections_truth_table(self):
        s=np.array([-2,2,-2,2.]);y=np.array([0,1,1,0])
        for dontcare_A in [0,1]:
            for dontcare_B in [0,1]:
                A=np.array([0,dontcare_A,1,0]);B=np.array([dontcare_B,0,0,1])
                np.testing.assert_array_equal((s+3*A-3*B>=0).astype(int),y)

    def test_interval_bound(self):
        rng=np.random.default_rng(17);s=rng.normal(size=10000);a=rng.uniform(0,4,10000);b=rng.uniform(0,4,10000)
        for A in [0,1]:
            for B in [0,1]:
                F=s+a*A-b*B
                self.assertTrue(np.all(F>=s-b-1e-12));self.assertTrue(np.all(F<=s+a+1e-12))
                self.assertTrue(np.all(F[s>=b]>=0));self.assertTrue(np.all(F[s < -a]<0))

    def test_numpy_torch_forward_agreement(self):
        torch.manual_seed(4);node=random_node();X=np.random.default_rng(2).normal(size=(30,4)).astype(np.float32)
        with torch.no_grad():expected=node.model(torch.tensor(X)).numpy().ravel()
        np.testing.assert_allclose(node.base_score(X),expected,rtol=1e-5,atol=1e-6)

    def test_recursive_certified_predictions(self):
        torch.manual_seed(3);node=random_node();X=np.random.default_rng(3).normal(size=(500,4)).astype(np.float32)
        pred,count=node.predict_certified(X)
        np.testing.assert_array_equal(pred,node.predict(X));self.assertLessEqual(count,len(X)*len(list(node.walk())))

    def test_multiclass_certified_predictions(self):
        torch.manual_seed(9);model=Forest(TrainConfig(),[2,4,8]);model.roots=[random_node() for _ in range(3)]
        X=np.random.default_rng(4).normal(size=(500,4)).astype(np.float32)
        pred,count=model.predict_certified(X)
        np.testing.assert_array_equal(pred,model.predict(X));self.assertLessEqual(count,len(X)*len(model.nodes()))

    def test_exact_zero_tie(self):
        n=Node(1,TrainConfig());n.model=make_mlp(1,1)
        with torch.no_grad():
            for p in n.model.parameters():p.zero_()
        X=np.zeros((3,1),dtype=np.float32)
        np.testing.assert_array_equal(n.predict_certified(X)[0],np.ones(3,dtype=int))

    def test_parameter_count(self):
        n=Node(4,TrainConfig(width=5));n.model=make_mlp(4,5)
        self.assertEqual(n.n_parameters(),5*(4+2)+1)

    def test_constrained_refit_corrects_known_errors(self):
        X=np.array([[-2.],[-1.],[1.],[2.]],dtype=np.float32);y=np.array([0,0,1,1])
        n=Node(1,TrainConfig(width=1));n.model=make_mlp(1,1)
        n.B=Node(1,TrainConfig(width=1));n.B.model=make_mlp(1,1)
        with torch.no_grad():
            for p in n.model.parameters():p.zero_()
            n.model[2].bias.fill_(.3)
            n.B.model[0].weight.fill_(-1);n.B.model[0].bias.zero_()
            n.B.model[2].weight.fill_(1);n.B.model[2].bias.fill_(-.5)
        n._refit(X,y,n.base_score(X))
        np.testing.assert_array_equal(n.predict(X),y)
        self.assertGreaterEqual(n.a,0);self.assertGreaterEqual(n.b,0)
        self.assertTrue(n.stats['refit_accepted'])

    def test_xor_can_interpolate(self):
        X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=np.float32);y=np.array([0,1,1,0])
        n=Node(2,TrainConfig(width=16,depth=0,epochs=400,lr=.03,batch_size=4,log_every=25)).fit(X,y)
        np.testing.assert_array_equal(n.predict(X),y)

    def test_training_acceptance_and_pruning(self):
        X,y=make_moons(n_samples=160,noise=.25,random_state=7);X=X.astype(np.float32)
        model=Forest(TrainConfig(width=4,depth=1,epochs=15,lr=.02,log_every=15),[0,1]).fit(X[:120],y[:120])
        for n in model.nodes():self.assertLessEqual(n.stats["final_errors"],n.stats["base_errors"])
        small,events=prune_forest(model,X[:120],y[:120],X[120:],y[120:])
        self.assertLessEqual(small.n_parameters(),model.n_parameters())
        self.assertGreaterEqual(np.mean(small.predict(X[:120])==y[:120]),np.mean(model.predict(X[:120])==y[:120]))
        self.assertGreaterEqual(np.mean(small.predict(X[120:])==y[120:])+0.005+1e-12,np.mean(model.predict(X[120:])==y[120:]))


if __name__=="__main__":unittest.main()
