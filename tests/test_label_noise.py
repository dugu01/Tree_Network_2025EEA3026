"""Supplementary NumPy checks; run separately from the original PyTorch tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec = importlib.util.spec_from_file_location('noise', Path(__file__).resolve().parents[1] / 'extras/label_noise_double_descent.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)

class NoiseTests(unittest.TestCase):
    def test_corruption(self):
        y = np.tile([0, 1], 100)
        a, idx = n.corrupt(y, 2, .15, 1234)
        b, idx2 = n.corrupt(y, 2, .15, 1234)
        self.assertEqual(len(idx), 30)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(idx, idx2)
        self.assertEqual(np.count_nonzero(a != y), 30)
        np.testing.assert_array_equal(y, np.tile([0, 1], 100))

    def check_gradient(self, k):
        rng = np.random.default_rng(80)
        x = rng.normal(size=(7, 3))
        y = np.arange(7) % (2 if k == 1 else k)
        weights = np.linspace(.6, 1.4, 7)
        model = n.MLP(3, 4, k, 17)
        model.p = [p.astype(float) for p in model.p]
        model.step(x, y, weights, lr=0)
        gradients = [m / .1 for m in model.m]
        def loss():
            z = model.logits(x)
            if k == 1:
                return np.mean(weights * (np.logaddexp(0, z[:, 0]) - y * z[:, 0]))
            z = z - z.max(axis=1, keepdims=True)
            return np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(7), y])
        for p, g in zip(model.p, gradients):
            numeric = np.zeros_like(p)
            for index in np.ndindex(p.shape):
                old = p[index]
                p[index] = old + 1e-6
                hi = loss()
                p[index] = old - 1e-6
                lo = loss()
                p[index] = old
                numeric[index] = (hi - lo) / 2e-6
            np.testing.assert_allclose(g, numeric, atol=1e-7, rtol=1e-5)
    def test_binary_gradient(self):
        self.check_gradient(1)
    def test_softmax_gradient(self):
        self.check_gradient(3)
    def test_invalid_noise(self):
        for frac in (-.1, 1, 1.1):
            with self.assertRaises(ValueError):
                n.corrupt(np.tile([0, 1], 100), 2, frac, 1)

if __name__ == '__main__':
    unittest.main()
