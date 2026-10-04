#!/usr/bin/env python3
"""Label-noise width sweep for the actual correction TREE (PyTorch, tree_model.py).

Answers two questions the root-only NumPy study cannot:
  1. Does the correction-tree architecture itself show an interpolation peak under
     label noise?  (root vs tree vs pruned test error across node width)
  2. What do the correction children learn when labels are noisy?  Children are
     trained on the parent's mistakes, so under label noise their targets are
     dominated by flipped labels.  We measure the share of flipped examples among
     each child's positive targets and among the training points the tree fixes,
     and test whether validation-only pruning removes the memorised noise.

Uses the unchanged experiment core (tree_model.Forest, prune_forest) and the same
corruption (noise seed 1234, 15%) as extras/label_noise_double_descent.py.
Validation/test labels stay clean.  Writes only to results_label_noise/.

  python extras/label_noise_tree.py sweep            # Cats vs Dogs, default widths/seeds
  python extras/label_noise_tree.py plot
  python extras/label_noise_tree.py sweep --dataset cifar10 --run-id mac_tree_cifar_noise15 --widths 1,2,4,8,16,32,64,128,256
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, copy, csv, importlib.util, json, sys, time, platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("lnoise", ROOT / "extras" / "label_noise_double_descent.py")
lnoise = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lnoise)

FIELDS = ["width", "seed", "kind", "nodes", "head_parameters", "noisy_train_errors", "clean_train_errors",
          "val_errors", "test_errors", "n_train", "n_val", "n_test", "test_err", "val_err", "test_logloss",
          "flipped_fit", "flipped_total", "seconds"]


def logloss(model, X, y):
    s = model.scores(X).astype(np.float64)
    if s.shape[1] == 1:
        s = s[:, 0]
        return float(np.mean(np.logaddexp(0, s) - y * s))
    s = s - s.max(1, keepdims=True)          # softmax over OVR scores (descriptive, as in run.py)
    return float(np.mean(np.log(np.exp(s).sum(1)) - s[np.arange(len(y)), y]))


def evaluate(model, arrays, yn, flipped_mask):
    X, y = arrays["train"]
    p = model.predict(X)
    out = dict(nodes=len(model.nodes()), head_parameters=model.n_parameters(),
               noisy_train_errors=int(np.sum(p != yn)), clean_train_errors=int(np.sum(p != y)),
               flipped_fit=int(np.sum((p == yn) & flipped_mask)), flipped_total=int(flipped_mask.sum()))
    for s in ["val", "test"]:
        q = model.predict(arrays[s][0])
        out[f"{s}_errors"] = int(np.sum(q != arrays[s][1]))
        out[f"{s}_err"] = out[f"{s}_errors"] / len(arrays[s][1])
    out["test_logloss"] = logloss(model, *arrays["test"])
    out.update(n_train=len(y), n_val=len(arrays["val"][1]), n_test=len(arrays["test"][1]))
    return out, p


def child_noise_shares(root_only, X, yn, flipped_mask):
    """Targets the roots' children receive (Table II groups), from the unrefit root MLPs.

    Binary: one root.  CIFAR-10: ten one-vs-rest roots; groups are pooled over roots
    (an example can appear in several roots' groups)."""
    C3s, C4s = [], []
    targets = [root_only.classes[1]] if len(root_only.classes) == 2 else list(root_only.classes)
    for r, k in zip(root_only.roots, targets):
        pred = r.base_score(X) >= 0
        local = yn == k
        C3s.append(~pred & local)    # false negatives -> child A positives
        C4s.append(pred & ~local)    # false positives -> child B positives
    C3, C4 = np.concatenate(C3s), np.concatenate(C4s)
    flipped_mask = np.tile(flipped_mask, len(targets))
    E = C3 | C4
    share = lambda m: float(flipped_mask[m].mean()) if m.any() else float("nan")
    return dict(root_errors=int(E.sum()), root_errors_flipped_share=share(E),
                childA_positives=int(C3.sum()), childA_flipped_share=share(C3),
                childB_positives=int(C4.sum()), childB_flipped_share=share(C4),
                base_rate=float(flipped_mask.mean()))


def do_sweep(a):
    import torch
    from tree_model import TrainConfig, Forest, prune_forest
    torch.set_num_threads(a.threads)
    cache = Path(a.cache) / a.dataset / "main"
    arrays, meta = lnoise.load(cache, verify=not a.no_verify)
    X, y = arrays["train"]
    k = int(y.max()) + 1
    classes = np.arange(k)
    yn, flipped = lnoise.corrupt(y, k, a.noise, a.noise_seed)
    fmask = np.zeros(len(y), bool); fmask[flipped] = True
    out = Path(a.out) / a.dataset / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    config = dict(dataset=a.dataset, noise=a.noise, noise_seed=a.noise_seed, epochs=a.epochs, depth=a.depth, widths=a.widths,
                  seeds=a.seeds, n_flipped=int(len(flipped)), cache_manifest_sha256=meta.get("manifest_sha256"),
                  pruning={"train_preserving": "max_train_drop=0, max_val_drop=0.005 (original rule)",
                           "validation_only": "max_train_drop=1, max_val_drop=0 (keep a child only if removing it lowers val accuracy)"},
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  platform=platform.platform(), script_sha256=lnoise.sha256(__file__))
    cpath = out / "config.json"
    if cpath.exists() and json.loads(cpath.read_text()) != config:
        raise ValueError("config differs from existing run; use a new --run-id")
    cpath.write_text(json.dumps(config, indent=2))
    csv_path = out / "tree_sweep.csv"
    done = set()
    if csv_path.exists():
        done = {(int(r["width"]), int(r["seed"])) for r in csv.DictReader(open(csv_path)) if r["kind"] == "pruned_val"}
    else:
        with open(csv_path, "w", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    Xv, yv = arrays["val"]
    for seed in a.seeds:
        for h in a.widths:
            if (h, seed) in done: continue
            rows, extra = [], {}
            t0 = time.perf_counter()
            root = Forest(TrainConfig(width=h, depth=0, epochs=a.epochs, seed=seed, fixed_epochs=True, log_every=10**9), classes).fit(X, yn)
            r, p_root = evaluate(root, arrays, yn, fmask); rows.append(dict(kind="root", seconds=time.perf_counter() - t0, **r))
            t0 = time.perf_counter()
            tree = Forest(TrainConfig(width=h, depth=a.depth, epochs=a.epochs, seed=seed, fixed_epochs=True, log_every=10**9), classes).fit(X, yn)
            r, p_tree = evaluate(tree, arrays, yn, fmask); rows.append(dict(kind="tree", seconds=time.perf_counter() - t0, **r))
            same_hidden = bool(all(np.array_equal(a_.model[0].weight.detach().numpy(), b_.model[0].weight.detach().numpy())
                                   for a_, b_ in zip(root.roots, tree.roots) if a_.model is not None and b_.model is not None))
            fixed = (p_root != yn) & (p_tree == yn)
            spoiled = (p_root == yn) & (p_tree != yn)
            extra.update(child_noise_shares(root, X, yn, fmask), root_tree_same_hidden_layer=same_hidden,
                         tree_fixed=int(fixed.sum()), tree_fixed_flipped_share=float(fmask[fixed].mean()) if fixed.any() else float("nan"),
                         tree_spoiled=int(spoiled.sum()),
                         nodes=[{k: v for k, v in n.stats.items() if k in ("name", "n", "positives", "base_errors", "final_errors", "refit_accepted", "fixed", "spoiled", "a", "b")} for n in tree.nodes()])
            for kind, kw in [("pruned_train", dict(max_train_drop=0., max_val_drop=.005)),
                             ("pruned_val", dict(max_train_drop=1., max_val_drop=0.))]:
                t0 = time.perf_counter()
                small, events = prune_forest(tree, X, yn, Xv, yv, **kw)
                r, _ = evaluate(small, arrays, yn, fmask); rows.append(dict(kind=kind, seconds=time.perf_counter() - t0, **r))
                extra[f"{kind}_events"] = events
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                for r in rows: w.writerow({"width": h, "seed": seed, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}})
            (out / f"analysis_w{h}_s{seed}.json").write_text(json.dumps(extra, indent=2))
            msg = "  ".join(f"{r['kind']}: test {100*r['test_err']:.2f}% noisy-tr {r['noisy_train_errors']} nodes {r['nodes']}" for r in rows)
            print(f"h={h} seed={seed} | {msg} | child-A flipped share {extra['childA_flipped_share']:.2f}, "
                  f"child-B {extra['childB_flipped_share']:.2f}, fixed-by-tree flipped share {extra['tree_fixed_flipped_share']:.2f}", flush=True)


def do_plot(a):
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(a.out) / a.dataset / a.run_id
    rows = list(csv.DictReader(open(out / "tree_sweep.csv")))
    widths = sorted({int(r["width"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    for w in widths:
        got = {int(r["seed"]) for r in rows if int(r["width"]) == w and r["kind"] == "pruned_val"}
        if got != set(seeds): raise ValueError(f"incomplete width {w}")
    def agg(kind, key, scale=100.):
        mu, sd = [], []
        for w in widths:
            v = np.array([float(r[key]) for r in rows if int(r["width"]) == w and r["kind"] == kind]) * scale
            mu.append(v.mean()); sd.append(v.std(ddof=1) if len(v) > 1 else 0.)
        return np.array(mu), np.array(sd)
    an = {(int(p.stem.split("_w")[1].split("_s")[0]), int(p.stem.split("_s")[-1])): json.loads(p.read_text()) for p in out.glob("analysis_w*_s*.json")}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.2), layout="constrained")
    style = {"root": ("Root only", "#365e87"), "tree": ("Tree", "#b46529"), "pruned_val": ("Tree, validation-pruned", "#4c9a5b")}
    for kind, (lab, c) in style.items():
        mu, sd = agg(kind, "test_err"); ax[0].errorbar(widths, mu, yerr=sd, marker="o", ms=3, capsize=2, lw=1.2, color=c, label=lab)
        n = int(rows[0]["n_train"])
        mu, sd = agg(kind, "noisy_train_errors", 100. / n); ax[1].errorbar(widths, mu, yerr=sd, marker="o", ms=3, capsize=2, lw=1.2, color=c, label=lab)
    for i, t in enumerate(["Test error (%)", "Noisy-train error (%)"]):
        ax[i].set_xscale("log", base=2); ax[i].set_xlabel("node width"); ax[i].set_ylabel(t); ax[i].grid(alpha=.18); ax[i].legend()
    ax[1].set_yscale("symlog", linthresh=.01)
    name = "Cats vs Dogs" if a.dataset == "catsdogs" else "CIFAR-10"
    ax[0].set_title(f"{name}, {round(100*json.loads((out/'config.json').read_text())['noise'])}% label noise"); ax[1].set_title("Fit to noisy training labels")
    for key, lab, c in [("childA_flipped_share", "child A positives", "#b46529"), ("childB_flipped_share", "child B positives", "#8a5aa8"), ("tree_fixed_flipped_share", "points fixed by tree", "#333333")]:
        mu = [np.nanmean([an[(w, s)][key] for s in seeds]) * 100 for w in widths]
        ax[2].plot(widths, mu, marker="o", ms=3, lw=1.2, color=c, label=lab)
    base = 100 * json.loads((out / "config.json").read_text())["noise"]
    ax[2].axhline(base, color="#999999", ls=":", lw=1, label=f"base rate ({base:.0f}%)")
    ax[2].set_xscale("log", base=2); ax[2].set_ylim(0, 105); ax[2].set_xlabel("node width"); ax[2].set_ylabel("share that are flipped labels (%)")
    ax[2].set_title("What the corrections learn"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.18)
    fig.savefig(out / "tree_noise.png"); print("wrote", out / "tree_noise.png")
    summ = []
    for w in widths:
        row = {"width": w}
        for kind in ["root", "tree", "pruned_train", "pruned_val"]:
            sel = [r for r in rows if int(r["width"]) == w and r["kind"] == kind]
            row[f"{kind}_test_err_pct"] = 100 * np.mean([float(r["test_err"]) for r in sel])
            row[f"{kind}_test_err_sd"] = 100 * np.std([float(r["test_err"]) for r in sel], ddof=1)
            row[f"{kind}_noisy_train_errors"] = "/".join(r["noisy_train_errors"] for r in sorted(sel, key=lambda r: int(r["seed"])))
            row[f"{kind}_nodes"] = "/".join(r["nodes"] for r in sorted(sel, key=lambda r: int(r["seed"])))
        for key in ["childA_flipped_share", "childB_flipped_share", "tree_fixed_flipped_share", "root_errors_flipped_share"]:
            row[key] = np.nanmean([an[(w, s)][key] for s in seeds])
        row["tree_fixed"] = "/".join(str(an[(w, s)]["tree_fixed"]) for s in seeds)
        summ.append(row)
    with open(out / "tree_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0])); wr.writeheader(); wr.writerows(summ)
    print("wrote", out / "tree_summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sweep", "plot"])
    p.add_argument("--dataset", choices=["catsdogs", "cifar10"], default="catsdogs")
    p.add_argument("--cache", default=str(ROOT / "cache"))
    p.add_argument("--out", default=str(ROOT / "results_label_noise"))
    p.add_argument("--run-id", default="tree_noise15_v1")
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=100, help="per learned node, as in the main tree sweep")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--widths", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512])
    p.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[17, 29, 43])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    if (Path(a.out).resolve() / "x").parent in [(ROOT / "results").resolve(), (ROOT / "provenance").resolve()]:
        raise ValueError("refusing to write into original evidence folders")
    {"sweep": do_sweep, "plot": do_plot}[a.command](a)


if __name__ == "__main__":
    main()
