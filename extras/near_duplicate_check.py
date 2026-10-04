#!/usr/bin/env python3
"""Find near-duplicate training images in the cached feature space (Cats vs Dogs).

Exact pixel hashing (data_features.py) cannot catch re-encoded or resized copies.
This lists training pairs whose standardized 576-d feature vectors are much closer
than typical neighbours, with their clean labels, their labels after the 15%
corruption (noise seed 1234) and their image paths.  A pair of near-identical
images that receives opposite noisy labels cannot be fitted by any model of
reasonable smoothness, which explains a single residual noisy-training mistake.
Read-only; writes results_label_noise/catsdogs/near_duplicates.json.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lnoise", ROOT / "extras" / "label_noise_double_descent.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)

cache = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache" / "catsdogs" / "main"
arrays, meta = L.load(cache)
X, y = arrays["train"]
yn, flipped = L.corrupt(y, 2, 0.15, 1234)
n2 = (X.astype(np.float64) ** 2).sum(1)
nn_d, nn_i = np.empty(len(X)), np.empty(len(X), int)
for i in range(0, len(X), 2000):
    D = n2[i:i + 2000, None] + n2[None, :] - 2 * X[i:i + 2000].astype(np.float64) @ X.T.astype(np.float64)
    D[np.arange(D.shape[0]), np.arange(i, i + D.shape[0])] = np.inf
    nn_i[i:i + 2000] = D.argmin(1); nn_d[i:i + 2000] = np.sqrt(np.maximum(D.min(1), 0))
median = float(np.median(nn_d))
manifest = json.loads((cache / "split_manifest.json").read_text())["train"]
pairs = sorted({tuple(sorted((int(i), int(nn_i[i])))) for i in np.where(nn_d < 0.1 * median)[0]})
out = {"median_nearest_neighbour_distance": median, "threshold": 0.1 * median, "pairs": []}
for a, b in pairs:
    out["pairs"].append({"indices": [a, b], "distance": float(np.linalg.norm(X[a] - X[b])),
                         "clean_labels": [int(y[a]), int(y[b])], "noisy_labels": [int(yn[a]), int(yn[b])],
                         "conflict_after_noise": bool(yn[a] != yn[b]),
                         "paths": [manifest[a].get("path", "").split("PetImages/")[-1], manifest[b].get("path", "").split("PetImages/")[-1]]})
dest = ROOT / "results_label_noise" / "catsdogs" / "near_duplicates.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
