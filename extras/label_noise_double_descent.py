#!/usr/bin/env python3
"""Label-noise width sweep (+ epoch-wise curve) on the cached MobileNetV3 features.

Purpose: test whether a classification-error double descent appears when the
training labels contain noise. Clean-label results in results/ are untouched.

Design (fixed topology, so width is the only capacity knob):
  * root-only one-hidden-layer ReLU MLP, no children, no refit, no pruning;
  * a fixed fraction of TRAIN labels is corrupted (fixed noise seed, shared by
    every width and optimisation seed); validation/test labels stay clean;
  * Adam, lr 1e-3, batch 512, class-balanced BCE (binary) as in the main runs,
    fixed number of epochs, no early stopping, no weight decay, no dropout;
  * test results are recorded for every width as exploratory descriptive results.
    Inspecting these curves can influence subsequent experiments; do not claim
    an untouched confirmatory test set for adaptively chosen follow-ups.

Binary (catsdogs): identical model family to the report's root nodes.
Multiclass (cifar10): ONE softmax MLP of hidden width h (not ten OVR roots) to
keep cost tractable; this differs from the report's CIFAR architecture.

Pure NumPy (float32) so it needs no torch/GPU. Writes only to --out
(default results_label_noise/), never to results/.

Examples (run from repo root):
  python extras/label_noise_double_descent.py sweep --dataset catsdogs
  python extras/label_noise_double_descent.py epochwise --dataset catsdogs --width 48
  python extras/label_noise_double_descent.py plot --dataset catsdogs
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time, platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTHS = {
    "catsdogs": [1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256, 512, 1024],
    "cifar10": [2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048],
}
FIELDS = ["dataset", "width", "seed", "noise", "epochs", "params", "noisy_train_err",
          "clean_train_err", "val_err", "test_err", "val_logloss", "test_logloss",
          "flipped", "n_train", "n_val", "n_test", "noisy_train_errors",
          "clean_train_errors", "val_errors", "test_errors", "seconds"]


# ---------------------------------------------------------------- data
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(cache, verify=True):
    cache = Path(cache)
    meta = json.loads((cache / "metadata.json").read_text())
    names = [f"{split}_{kind}.npy" for split in ("train", "val", "test") for kind in ("X", "y")]
    if verify:
        hashes = meta.get("file_hashes", {})
        for name in names:
            if name not in hashes or sha256(cache / name) != hashes[name]:
                raise ValueError(f"missing hash or changed cache file: {name}")
        manifest = cache / "split_manifest.json"
        if not manifest.is_file() or sha256(manifest) != meta.get("manifest_sha256"):
            raise ValueError("split manifest missing or changed")
    a = {s: (np.load(cache / f"{s}_X.npy", allow_pickle=False),
             np.load(cache / f"{s}_y.npy", allow_pickle=False)) for s in ["train", "val", "test"]}
    d = a["train"][0].shape[1]
    classes = np.unique(a["train"][1])
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
        raise ValueError("training labels must contain contiguous classes starting at zero")
    for split, (x, y) in a.items():
        if x.ndim != 2 or x.shape[1] != d or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError(f"invalid cache shapes: {split}")
        if not np.isfinite(x).all() or not np.isin(y, classes).all():
            raise ValueError(f"nonfinite features or invalid labels: {split}")
    mean = a["train"][0].mean(0)
    std = np.maximum(a["train"][0].std(0), 1e-5)          # same rule as data_features.load_features
    a = {s: (((x - mean) / std).astype(np.float32), y.astype(np.int64)) for s, (x, y) in a.items()}
    return a, meta


def corrupt(y, n_classes, frac, seed):
    """Flip exactly round(frac*N) labels (uniform over examples); multiclass flips go to a uniformly random other class."""
    if not 0 <= frac < 1:
        raise ValueError("noise fraction must be in [0,1)")
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.choice(n, size=int(round(frac * n)), replace=False)
    noisy = y.copy()
    if n_classes == 2:
        noisy[idx] = 1 - y[idx]
    else:
        noisy[idx] = (y[idx] + rng.integers(1, n_classes, size=len(idx))) % n_classes
    return noisy, idx


# ---------------------------------------------------------------- model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class MLP:
    """d -> h (ReLU) -> k outputs; k=1 logit (binary) or k=n_classes (softmax). PyTorch-style init."""
    def __init__(self, d, h, k, seed):
        r = np.random.default_rng(seed)
        b1, b2 = 1 / np.sqrt(d), 1 / np.sqrt(h)
        f = np.float32
        self.p = [r.uniform(-b1, b1, (d, h)).astype(f), r.uniform(-b1, b1, h).astype(f),
                  r.uniform(-b2, b2, (h, k)).astype(f), r.uniform(-b2, b2, k).astype(f)]
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0
        self.k = k

    def n_params(self):
        return int(sum(x.size for x in self.p))

    def logits(self, X, chunk=8192):
        W1, c1, W2, c2 = self.p
        out = [np.maximum(X[i:i + chunk] @ W1 + c1, 0) @ W2 + c2 for i in range(0, len(X), chunk)]
        return np.concatenate(out)

    def step(self, Xb, yb, wb, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        W1, c1, W2, c2 = self.p
        Z = Xb @ W1 + c1
        R = np.maximum(Z, 0)
        S = R @ W2 + c2
        B = len(yb)
        if self.k == 1:
            dS = ((sigmoid(S[:, 0]) - yb) * wb / B)[:, None]
        else:
            S = S - S.max(1, keepdims=True)
            P = np.exp(S); P /= P.sum(1, keepdims=True)
            P[np.arange(B), yb] -= 1
            dS = P / B
        dW2 = R.T @ dS
        dc2 = dS.sum(0)
        dZ = (dS @ W2.T) * (Z > 0)
        grads = [Xb.T @ dZ, dZ.sum(0), dW2, dc2]
        if not all(np.isfinite(g).all() for g in grads):
            raise FloatingPointError("nonfinite gradient; aborting run")
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            self.p[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


def predict(model, X):
    s = model.logits(X)
    return (s[:, 0] >= 0).astype(np.int64) if model.k == 1 else s.argmax(1)


def logloss(model, X, y):
    s = model.logits(X).astype(np.float64)
    if model.k == 1:
        s = s[:, 0]
        return float(np.mean(np.logaddexp(0, s) - y * s))
    s -= s.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(s).sum(1)) - s[np.arange(len(y)), y]))


def train(model, X, y_noisy, epochs, seed, batch=512, lr=1e-3, callback=None, every=1):
    n = len(y_noisy)
    if model.k == 1:
        n1 = int(y_noisy.sum()); n0 = n - n1
        if not n0 or not n1:
            raise ValueError("corrupted labels must retain both classes")
        w = np.where(y_noisy > 0, n / (2 * n1), n / (2 * n0)).astype(np.float32)
        yf = y_noisy.astype(np.float32)
    else:
        w, yf = None, y_noisy
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for j in range(0, n, batch):
            i = perm[j:j + batch]
            model.step(X[i], yf[i], None if w is None else w[i], lr=lr)
        if callback is not None and (ep == 1 or ep % every == 0 or ep == epochs):
            callback(ep, model)


# ---------------------------------------------------------------- experiments
def setup(args):
    cache = Path(args.cache) / args.dataset / args.profile if args.cache_is_root else Path(args.cache)
    arrays, meta = load(cache, verify=not args.no_verify)
    k = int(max(arrays["train"][1].max(), 1) + 1)
    yn, flipped = corrupt(arrays["train"][1], k, args.noise, args.noise_seed)
    return arrays, meta, k, yn, flipped


def out_dir(args):
    base = Path(args.out).resolve()
    protected = [(ROOT / "results").resolve(), (ROOT / "provenance").resolve()]
    if any(base == q or q in base.parents for q in protected):
        raise ValueError("output must not overwrite original evidence")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    p = base / args.dataset / args.run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_contract(args, arrays, meta, yn, flipped, name):
    out = out_dir(args)
    config = {"dataset": args.dataset, "profile": args.profile, "noise": args.noise,
        "noise_seed": args.noise_seed, "epochs": args.epochs,
        "widths": args.widths if args.command == "sweep" else [args.width],
        "seeds": args.seeds, "batch": args.batch, "lr": args.lr,
        "every": args.every if args.command == "epochwise" else None,
        "n_flipped": len(flipped), "n_samples": {k:len(v[1]) for k,v in arrays.items()},
        "cache_manifest_sha256": meta.get("manifest_sha256"),
        "cache_file_hashes": meta.get("file_hashes"), "verified_cache": not args.no_verify,
        "script_sha256": sha256(__file__), "python": platform.python_version(),
        "numpy": np.__version__, "noise_indices_sha256": hashlib.sha256(flipped.tobytes()).hexdigest(),
        "model": "binary root MLP" if len(np.unique(yn)) == 2 else "single softmax MLP",
        "evaluation_scope": "exploratory; no claim of untouched confirmatory test evaluation"}
    path = out / (name + "_config.json")
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("run contract differs; choose a new --run-id (no results were overwritten)")
    else:
        if (out / (name + ".csv")).exists():
            raise ValueError("CSV exists without provenance; use a new --run-id")
        path.write_text(json.dumps(config, indent=2))
        np.savez_compressed(out / (name + "_corruption.npz"),
                            flipped_indices=flipped, clean_labels=arrays["train"][1], noisy_labels=yn)
    return out


def do_sweep(args):
    arrays, meta, k, yn, flipped = setup(args)
    X, y = arrays["train"]
    d = X.shape[1]
    tag = f"sweep_noise{int(round(args.noise*100))}"
    out = ensure_contract(args, arrays, meta, yn, flipped, tag)
    csv_path = out / (tag + ".csv")
    done = set()
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            done.add((int(r["width"]), int(r["seed"])))
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for seed in args.seeds:
        for h in args.widths:
            if (h, seed) in done:
                continue
            t0 = time.perf_counter()
            m = MLP(d, h, 1 if k == 2 else k, seed)
            train(m, X, yn, args.epochs, seed, batch=args.batch, lr=args.lr)
            preds = {s: predict(m, v[0]) for s,v in arrays.items()}
            row = dict(dataset=args.dataset, width=h, seed=seed, noise=args.noise, epochs=args.epochs,
                       params=m.n_params(), noisy_train_err=float(np.mean(predict(m, X) != yn)),
                       clean_train_err=float(np.mean(predict(m, X) != y)),
                       val_err=float(np.mean(predict(m, arrays['val'][0]) != arrays['val'][1])),
                       test_err=float(np.mean(predict(m, arrays['test'][0]) != arrays['test'][1])),
                       val_logloss=logloss(m, *arrays['val']), test_logloss=logloss(m, *arrays['test']),
                       flipped=int(len(flipped)), n_train=len(y),
                       n_val=len(arrays["val"][1]), n_test=len(arrays["test"][1]),
                       noisy_train_errors=int(np.sum(preds["train"] != yn)),
                       clean_train_errors=int(np.sum(preds["train"] != y)),
                       val_errors=int(np.sum(preds["val"] != arrays["val"][1])),
                       test_errors=int(np.sum(preds["test"] != arrays["test"][1])), seconds=round(time.perf_counter() - t0, 1))
            stem = f"{tag}_w{h}_s{seed}"
            np.savez_compressed(out / (stem + "_predictions.npz"), **preds)
            (out / (stem + "_metrics.json")).write_text(json.dumps(row, indent=2))
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"{args.dataset} h={h:5d} seed={seed}: noisy-train {100*row['noisy_train_err']:.2f}%  "
                  f"val {100*row['val_err']:.2f}%  test {100*row['test_err']:.2f}%  ({row['seconds']}s)", flush=True)


def do_epochwise(args):
    arrays, meta, k, yn, flipped = setup(args)
    X, y = arrays["train"]
    tag = f"epochwise_w{args.width}_noise{int(round(args.noise*100))}"
    out = ensure_contract(args, arrays, meta, yn, flipped, tag)
    path = out / (tag + ".csv")
    if path.exists():
        print(f"epoch-wise run already complete: {path} (skipping)")
        return
    rows = []
    for seed in args.seeds:
        m = MLP(X.shape[1], args.width, 1 if k == 2 else k, seed)
        def cb(ep, model, seed=seed):
            rows.append(dict(seed=seed, epoch=ep,
                             clean_train_err=float(np.mean(predict(model, X) != y)),
                             noisy_train_errors=int(np.sum(predict(model, X) != yn)),
                             n_train=len(y), n_val=len(arrays["val"][1]), n_test=len(arrays["test"][1]),
                             noisy_train_err=float(np.mean(predict(model, X) != yn)),
                             val_err=float(np.mean(predict(model, arrays['val'][0]) != arrays['val'][1])),
                             test_err=float(np.mean(predict(model, arrays['test'][0]) != arrays['test'][1])),
                             test_logloss=logloss(model, *arrays['test'])))
            if ep % 50 == 0:
                print(f"seed {seed} epoch {ep}: noisy-train {100*rows[-1]['noisy_train_err']:.2f}%  test {100*rows[-1]['test_err']:.2f}%", flush=True)
        train(m, X, yn, args.epochs, seed, batch=args.batch, lr=args.lr, callback=cb, every=args.every)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", path)


def do_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = out_dir(args)
    tag = f"noise{int(round(args.noise*100))}"
    sweep = list(csv.DictReader(open(out / f"sweep_{tag}.csv")))
    widths = sorted({int(r["width"]) for r in sweep})
    expected = set(args.seeds)
    for width in widths:
        seeds = [int(r["seed"]) for r in sweep if int(r["width"]) == width]
        if len(seeds) != len(set(seeds)) or set(seeds) != expected:
            raise ValueError("incomplete/duplicate seeds; finish sweep or specify matching --seeds")
    summary=[]
    for width in widths:
        rows=[r for r in sweep if int(r["width"]) == width]
        values=np.array([float(r["test_err"])*100 for r in rows])
        summary.append(dict(width=width, params=int(rows[0]["params"]), n_seeds=len(rows),
          noisy_train_err_pct=np.mean([float(r["noisy_train_err"])*100 for r in rows]),
          val_err_pct=np.mean([float(r["val_err"])*100 for r in rows]),
          test_err_pct_mean=values.mean(), test_err_pct_sd=values.std(ddof=1) if len(rows)>1 else 0.,
          test_logloss_mean=np.mean([float(r["test_logloss"]) for r in rows])))
    with open(out/f"summary_{tag}.csv", "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    def agg(key, scale=100.0):
        mu, sd = [], []
        for w in widths:
            v = np.array([float(r[key]) for r in sweep if int(r["width"]) == w]) * scale
            mu.append(v.mean()); sd.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        return np.array(mu), np.array(sd)
    ep_files = list(out.glob(f"epochwise_w{args.width}_{tag}.csv"))
    ncol = 3 if ep_files else 2
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
    fig, ax = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.2), layout="constrained")
    nz, _ = agg("noisy_train_err")
    interp = next((w for w, e in zip(widths, nz) if e == 0.0), None)  # exact zero in every completed seed
    for key, lab, c in [("noisy_train_err", "train (noisy labels)", "#999999"), ("val_err", "validation", "#365e87"), ("test_err", "test", "#b46529")]:
        mu, sd = agg(key)
        ax[0].errorbar(widths, mu, yerr=sd, marker="o", ms=3, capsize=2, lw=1.2, label=lab, color=c)
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("hidden width"); ax[0].set_ylabel("error (%)")
    ax[0].set_title(f"{args.dataset}: error vs width ({tag})"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.18)
    for key, lab, c in [("val_logloss", "validation", "#365e87"), ("test_logloss", "test", "#b46529")]:
        mu, sd = agg(key, 1.0)
        ax[1].errorbar(widths, mu, yerr=sd, marker="o", ms=3, capsize=2, lw=1.2, label=lab, color=c)
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("hidden width"); ax[1].set_ylabel("log loss"); ax[1].set_title("log loss vs width")
    ax[1].legend(); ax[1].grid(alpha=.18)
    if interp:
        for a in ax[:2]:
            a.axvline(interp, color="#777777", ls=":", lw=1)
    if ep_files:
        rows = list(csv.DictReader(open(ep_files[0])))
        eps = sorted({int(r["epoch"]) for r in rows})
        for key, lab, c in [("noisy_train_err", "train (noisy)", "#999999"), ("test_err", "test", "#b46529")]:
            mu = [np.mean([float(r[key]) for r in rows if int(r["epoch"]) == e]) * 100 for e in eps]
            ax[2].plot(eps, mu, label=lab, color=c, lw=1.2)
        ax[2].set_xscale("log"); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("error (%)")
        ax[2].set_title("epoch-wise, " + ep_files[0].stem.split("_")[1]); ax[2].legend(); ax[2].grid(alpha=.18)
    fig.savefig(out / f"noise_double_descent_{tag}.png")
    print("wrote", out / f"noise_double_descent_{tag}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sweep", "epochwise", "plot"])
    p.add_argument("--dataset", choices=["catsdogs", "cifar10"], default="catsdogs")
    p.add_argument("--cache", default=str(ROOT / "cache"), help="cache root (contains <dataset>/<profile>) or a direct cache dir with --direct-cache")
    p.add_argument("--direct-cache", dest="cache_is_root", action="store_false", help="treat --cache as the cache directory itself")
    p.add_argument("--profile", default="main")
    p.add_argument("--run-id", default="verified_rerun_v1", help="isolated immutable run configuration")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=str(ROOT / "results_label_noise"))
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--widths", type=lambda s: [int(v) for v in s.split(",")], default=None)
    p.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[17, 29, 43])
    p.add_argument("--width", type=int, default=48, help="width for epochwise")
    p.add_argument("--every", type=int, default=5, help="epochwise logging interval")
    p.add_argument("--no-verify", action="store_true", help="skip cache hash verification")
    a = p.parse_args()
    a.widths = a.widths or DEFAULT_WIDTHS[a.dataset]
    if not 0 <= a.noise < 1 or min(a.epochs,a.every,a.width,a.batch,*a.widths) < 1 or a.lr <= 0:
        p.error("invalid noise, width, epochs, interval, batch or learning rate")
    if not a.seeds or len(a.seeds)!=len(set(a.seeds)) or len(a.widths)!=len(set(a.widths)):
        p.error("seeds and widths must be nonempty and unique")
    {"sweep": do_sweep, "epochwise": do_epochwise, "plot": do_plot}[a.command](a)


if __name__ == "__main__":
    main()
