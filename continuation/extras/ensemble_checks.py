"""Recompute seven-member ensemble diagnostics from saved test probabilities.

Run from the continuation folder:  python extras/ensemble_checks.py
Reports, per dataset and family: mean member accuracy, hard-vote accuracy,
vote ties, score-sum tie-break accuracy, pairwise disagreement and double fault,
plus paired image-bootstrap intervals (10,000 resamples, seed 2026) for the
family differences quoted in Section 4 of the report.
"""
import glob, itertools, json
import numpy as np

def family_stats(P, y):
    K = P.shape[-1]; L = P.argmax(-1)
    counts = (L[..., None] == np.arange(K)).sum(0)
    mx = counts.max(-1, keepdims=True); tie = (counts == mx).sum(-1) > 1
    vote = counts.argmax(-1)                                  # smallest class index on ties
    vote_ss = np.where(counts == mx, P.sum(0), -np.inf).argmax(-1)  # score-sum tie-break
    pairs = list(itertools.combinations(range(len(L)), 2))
    return dict(member_acc=100 * (L == y).mean(),
                vote_acc=100 * (vote == y).mean(), ties=int(tie.sum()),
                scoresum_acc=100 * (vote_ss == y).mean(),
                disagreement=100 * np.mean([(L[i] != L[j]).mean() for i, j in pairs]),
                double_fault=100 * np.mean([((L[i] != y) & (L[j] != y)).mean() for i, j in pairs]),
                _v=(vote == y), _vss=(vote_ss == y))

for ds in ["catsdogs", "cifar10"]:
    y = np.load(f"banks/{ds}/test_y.npy").astype(int)
    fam = {}
    for f in glob.glob(f"runs/{ds}/ensemble_*/request.json"):
        job = json.load(open(f))["job"]
        P = np.load(f.replace("request.json", "test_predictions.npz"))["test"].astype(float)
        fam.setdefault(job["ensemble_family"], []).append((job["seed"], P))
    out = {k: family_stats(np.stack([p for _, p in sorted(v, key=lambda t: t[0])]), y) for k, v in fam.items()}
    print(f"\n== {ds} (n={len(y)})")
    for k, o in sorted(out.items()):
        print(f"{k:20s} " + "  ".join(f"{m}={o[m]:.2f}" if isinstance(o[m], float) else f"{m}={o[m]}"
                                     for m in ["member_acc", "vote_acc", "ties", "scoresum_acc", "disagreement", "double_fault"]))
    rng = np.random.default_rng(2026)
    for a, b in [("augment_diverse", "none"), ("persistent_diverse", "augment_diverse")]:
        for key, name in [("_v", "vote"), ("_vss", "score-sum")]:
            d = out[a][key].astype(float) - out[b][key].astype(float)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) * 100
            print(f"{a} - {b} [{name}]: {100*d.mean():+.2f} points, 95% CI [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
