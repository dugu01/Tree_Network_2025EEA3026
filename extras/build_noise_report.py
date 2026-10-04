#!/usr/bin/env python3
"""Build the label-noise figures/tables for the report from results_label_noise/ (no training).

Checks that every expected width/seed record exists, that the configurations match
the report (15% noise, fixed noise seed, verified cache), and writes
  report/figures/noise_root_curves.png, report/figures/noise_tree_curves.png,
  report/tables/noise_root_rows.tex, report/tables/noise_tree_rows.tex,
  provenance/label_noise_audit.json
"""
from pathlib import Path
import csv, json
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_label_noise"
FIG, TAB = ROOT / "report" / "figures", ROOT / "report" / "tables"
SEEDS = [17, 29, 43]
NTR = {"catsdogs": 17476, "cifar10": 45000}
audit = {"checks": [], "anomalies": []}


def read(path):
    return list(csv.DictReader(open(path)))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    audit["checks"].append(msg)


def sweep(ds, run, noise):
    tag = f"noise{round(noise * 100)}"
    d = R / ds / run
    cfg = json.loads((d / f"sweep_{tag}_config.json").read_text())
    rows = read(d / f"sweep_{tag}.csv")
    widths = cfg["widths"]
    got = {(int(r["width"]), int(r["seed"])) for r in rows}
    check(got == {(w, s) for w in widths for s in SEEDS} and len(rows) == len(widths) * 3,
          f"{ds}/{run}: complete {len(widths)} widths x 3 seeds")
    check(cfg["verified_cache"] and abs(cfg["noise"] - noise) < 1e-12 and cfg["epochs"] == 300,
          f"{ds}/{run}: noise {noise}, 300 epochs, verified cache")
    out = {}
    for w in widths:
        a = [r for r in rows if int(r["width"]) == w]
        g = lambda k: np.array([float(r[k]) for r in a])
        n = int(a[0]["n_train"]) if "n_train" in a[0] else NTR[ds]
        nte = g("test_err")
        out[w] = dict(params=int(a[0]["params"]), noisy=100 * g("noisy_train_err").mean(), noisy_counts=[round(float(r["noisy_train_err"]) * n) for r in sorted(a, key=lambda r: int(r["seed"]))],
                      val=100 * g("val_err").mean(), test=100 * nte.mean(), test_sd=100 * nte.std(ddof=1), ll=g("test_logloss").mean())
    # flag convergence anomalies: noisy-train error far above both neighbours' max
    ws = sorted(out)
    for i, w in enumerate(ws):
        if 0 < i < len(ws) - 1:
            nb = max(max(out[ws[i - 1]]["noisy_counts"]), max(out[ws[i + 1]]["noisy_counts"]))
            for s, c in zip(SEEDS, out[w]["noisy_counts"]):
                if c > 10 * max(nb, 5):
                    audit["anomalies"].append(f"{ds}/{run}: width {w} seed {s} ended with {c} noisy-train errors (neighbours <= {nb})")
    w = ws[-1]
    for s, c in zip(SEEDS, out[w]["noisy_counts"]):
        if c > 10 * max(max(out[ws[-2]]["noisy_counts"]), 5):
            audit["anomalies"].append(f"{ds}/{run}: width {w} seed {s} ended with {c} noisy-train errors (previous width <= {max(out[ws[-2]]['noisy_counts'])})")
    return out, cfg


def interp_width(curve, tol=0.05):
    return next((w for w in sorted(curve) if curve[w]["noisy"] < tol), None)


def tree(ds, run):
    d = R / ds / run
    cfg = json.loads((d / "config.json").read_text())
    rows = read(d / "tree_sweep.csv")
    widths = cfg["widths"]
    for kind in ["root", "tree", "pruned_train", "pruned_val"]:
        got = {(int(r["width"]), int(r["seed"])) for r in rows if r["kind"] == kind}
        check(got == {(w, s) for w in widths for s in SEEDS}, f"{ds}/{run}: complete {kind} records")
    check(abs(cfg["noise"] - 0.15) < 1e-12 and cfg["noise_seed"] == 1234 and cfg["epochs"] == 100 and cfg["depth"] == 2,
          f"{ds}/{run}: 15% noise, seed 1234, 100 epochs/node, depth 2")
    out = {}
    for w in widths:
        e = {}
        for kind in ["root", "tree", "pruned_train", "pruned_val"]:
            a = sorted([r for r in rows if int(r["width"]) == w and r["kind"] == kind], key=lambda r: int(r["seed"]))
            e[kind] = dict(test=100 * np.mean([float(r["test_err"]) for r in a]), sd=100 * np.std([float(r["test_err"]) for r in a], ddof=1),
                           noisy=[int(r["noisy_train_errors"]) for r in a], nodes=[int(r["nodes"]) for r in a])
        an = [json.loads((d / f"analysis_w{w}_s{s}.json").read_text()) for s in SEEDS]
        check(all(x["root_tree_same_hidden_layer"] for x in an), f"{ds} width {w}: tree root hidden layer identical to root-only model")
        e["A"] = np.nanmean([x["childA_flipped_share"] for x in an]) * 100
        e["B"] = np.nanmean([x["childB_flipped_share"] for x in an]) * 100
        e["fixed_share"] = np.nanmean([x["tree_fixed_flipped_share"] for x in an]) * 100
        e["fixed"] = [x["tree_fixed"] for x in an]
        e["root_err_share"] = np.nanmean([x["root_errors_flipped_share"] for x in an]) * 100
        out[w] = e
    for w in widths:
        for s, c in zip(SEEDS, out[w]["root"]["noisy"]):
            ws = [v for v in widths if v < w]
            if ws and c > 10 * max(5, max(out[ws[-1]]["root"]["noisy"]) if w > 32 else 10**9):
                audit["anomalies"].append(f"{ds}/{run}: root-only width {w} seed {s} ended with {c} noisy-train errors")
    return out, cfg


plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "savefig.dpi": 220})
C = {"n15": "#b46529", "n0": "#365e87", "train": "#888888", "tree": "#b46529", "root": "#365e87", "pv": "#4c9a5b"}

# ---------------- root-only curves
cd15, _ = sweep("catsdogs", "mac_noise15", .15)
cd0, _ = sweep("catsdogs", "mac_noise0", 0.)
cdA, _ = sweep("catsdogs", "mac_noise15_ns5678", .15)
cdB, _ = sweep("catsdogs", "mac_noise15_ns9012", .15)
cf15, _ = sweep("cifar10", "mac_cifar_noise15", .15)
cf0, _ = sweep("cifar10", "mac_cifar_noise0", 0.)
ep = read(R / "catsdogs" / "mac_noise15" / "epochwise_w16_noise15.csv")

fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.3), layout="constrained")
def curve(a, data, key, label, color, sd=True, **kw):
    ws = sorted(data)
    y = [data[w][key] for w in ws]
    if sd and key == "test":
        a.errorbar(ws, y, yerr=[data[w]["test_sd"] for w in ws], marker="o", ms=2.5, capsize=1.5, lw=1.2, color=color, label=label, **kw)
    else:
        a.plot(ws, y, marker="o", ms=2, lw=1, color=color, label=label, **kw)
curve(ax[0], cd15, "test", "test, 15% noise", C["n15"])
curve(ax[0], cdA, "test", "test, 15% noise (other noise draws)", C["n15"], sd=False, alpha=.45, ls="--")
curve(ax[0], cdB, "test", None, C["n15"], sd=False, alpha=.45, ls="--")
curve(ax[0], cd0, "test", "test, clean labels", C["n0"])
curve(ax[0], cd15, "noisy", "train error on noisy labels", C["train"], sd=False, ls=":")
iw = interp_width(cd15); ax[0].axvline(iw, color="#777", ls=":", lw=.8)
ax[0].set_title("Cats vs Dogs: root-only MLP (300 epochs)")
curve(ax[1], cf15, "test", "test, 15% noise", C["n15"])
curve(ax[1], cf0, "test", "test, clean labels", C["n0"])
curve(ax[1], cf15, "noisy", "train error on noisy labels", C["train"], sd=False, ls=":")
curve(ax[1], cf0, "noisy", "train error, clean labels", C["n0"], sd=False, ls=":", alpha=.6)
iwc = interp_width(cf15); ax[1].axvline(iwc, color="#777", ls=":", lw=.8)
ax[1].set_title("CIFAR-10: softmax MLP (300 epochs)")
for a in ax[:2]:
    a.set_xscale("log", base=2); a.set_xlabel("hidden width"); a.set_ylabel("error (%)"); a.grid(alpha=.18); a.legend(fontsize=6.5)
eps = sorted({int(r["epoch"]) for r in ep})
for key, lab, c in [("noisy_train_err", "train (noisy labels)", C["train"]), ("test_err", "test", C["n15"])]:
    ax[2].plot(eps, [100 * np.mean([float(r[key]) for r in ep if int(r["epoch"]) == e]) for e in eps], color=c, lw=1.2, label=lab)
ax[2].set_xscale("log"); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("error (%)"); ax[2].grid(alpha=.18); ax[2].legend(fontsize=6.5)
ax[2].set_title("Cats vs Dogs: epoch-wise, width 16, 15% noise")
fig.savefig(FIG / "noise_root_curves.png"); plt.close(fig)

# ---------------- tree curves
tcd, _ = tree("catsdogs", "mac_tree_noise15")
tcf, _ = tree("cifar10", "mac_tree_cifar_noise15")
fig, ax = plt.subplots(2, 3, figsize=(10.8, 6.0), layout="constrained")
for row, (ds, T) in enumerate([("Cats vs Dogs", tcd), ("CIFAR-10", tcf)]):
    ws = sorted(T); n = NTR["catsdogs" if row == 0 else "cifar10"]
    for kind, lab, c in [("root", "root only", C["root"]), ("tree", "tree", C["tree"]), ("pruned_val", "tree, validation-pruned", C["pv"])]:
        ax[row, 0].errorbar(ws, [T[w][kind]["test"] for w in ws], yerr=[T[w][kind]["sd"] for w in ws], marker="o", ms=2.5, capsize=1.5, lw=1.2, color=c, label=lab)
    for kind, lab, c in [("root", "root only", C["root"]), ("tree", "tree", C["tree"])]:
        ax[row, 1].plot(ws, [100 * np.median(T[w][kind]["noisy"]) / n for w in ws], marker="o", ms=2.5, lw=1.2, color=c, label=lab)
    ax[row, 1].set_yscale("symlog", linthresh=.01); ax[row, 1].set_ylim(bottom=0)
    sel = [w for w in ws if min(T[w]["root"]["noisy"]) >= 100]
    for key, lab, c in [("A", "child A positives", C["tree"]), ("B", "child B positives", "#8a5aa8"), ("fixed_share", "training points fixed by tree", "#333333")]:
        ax[row, 2].plot(sel, [T[w][key] for w in sel], marker="o", ms=2.5, lw=1.2, color=c, label=lab)
    ax[row, 2].axhline(15, color="#999", ls=":", lw=1, label="noise rate (15%)"); ax[row, 2].set_ylim(0, 100)
    ax[row, 0].set_title(f"{ds}: test error, 15% noise"); ax[row, 1].set_title(f"{ds}: train error on noisy labels (median)")
    ax[row, 2].set_title(f"{ds}: share that are flipped labels")
    ax[row, 0].set_ylabel("error (%)"); ax[row, 1].set_ylabel("error (%)"); ax[row, 2].set_ylabel("flipped (%)")
    for a in ax[row]:
        a.set_xscale("log", base=2); a.set_xlabel("node width"); a.grid(alpha=.18); a.legend(fontsize=6.5)
fig.savefig(FIG / "noise_tree_curves.png"); plt.close(fig)

# ---------------- tables
def f2(x): return f"{x:.2f}"
lines = []
for name, data, clean, sel in [("C/D", cd15, cd0, [1, 4, 8, 12, 16, 20, 24, 32, 64, 128, 256, 1024]),
                               ("C10", cf15, cf0, [2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024])]:
    best = min(data, key=lambda w: data[w]["test"]) ; peak = max([w for w in data if w > best], key=lambda w: data[w]["test"])
    for w in sel:
        d = data[w]; t = f"{d['test']:.2f} ({d['test_sd']:.2f})"
        if w in (best, peak): t = r"\textbf{" + f"{d['test']:.2f}" + "}" + f" ({d['test_sd']:.2f})"
        lines.append(f"{name} & {w} & {d['params']:,} & {f2(d['noisy'])} & {t} & {f2(clean[w]['noisy'])} & {f2(clean[w]['test'])} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
(TAB / "noise_root_rows.tex").write_text("\n".join(lines) + "\n")
lines = []
for name, T, sel in [("C/D", tcd, [1, 4, 8, 12, 16, 24, 32, 64, 128, 512]), ("C10", tcf, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])]:
    for w in sel:
        e = T[w]; sh = (f"{e['A']:.0f} / {e['fixed_share']:.0f}" if min(e["root"]["noisy"]) >= 100 else "--")
        lines.append(f"{name} & {w} & {e['root']['test']:.2f} & {e['tree']['test']:.2f} & {e['pruned_train']['test']:.2f} & {e['pruned_val']['test']:.2f} & "
                     f"{round(np.mean(e['root']['noisy'])):,} & {round(np.mean(e['tree']['noisy'])):,} & {sh} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
(TAB / "noise_tree_rows.tex").write_text("\n".join(lines) + "\n")

key = {
    "catsdogs_root": {"interp_width": iw, "best": min(cd15.items(), key=lambda kv: kv[1]["test"])[0],
                      "test_by_width": {w: round(v["test"], 2) for w, v in cd15.items()},
                      "other_noise_draws_test_by_width": [{w: round(v["test"], 2) for w, v in x.items()} for x in (cdA, cdB)],
                      "clean_test_by_width": {w: round(v["test"], 2) for w, v in cd0.items()}},
    "cifar10_root": {"interp_width": iwc, "test_by_width": {w: round(v["test"], 2) for w, v in cf15.items()},
                     "clean_test_by_width": {w: round(v["test"], 2) for w, v in cf0.items()},
                     "clean_noisy_train_by_width": {w: round(v["noisy"], 3) for w, v in cf0.items()}},
    "trees": {ds: {w: {"root": round(e["root"]["test"], 2), "tree": round(e["tree"]["test"], 2), "pruned_train": round(e["pruned_train"]["test"], 2),
                       "pruned_val": round(e["pruned_val"]["test"], 2), "root_noisy": e["root"]["noisy"], "tree_noisy": e["tree"]["noisy"],
                       "nodes_tree": e["tree"]["nodes"], "nodes_pruned_val": e["pruned_val"]["nodes"], "childA_flipped_pct": round(e["A"], 1),
                       "childB_flipped_pct": round(e["B"], 1), "fixed_flipped_pct": round(e["fixed_share"], 1), "fixed": e["fixed"],
                       "root_errors_flipped_pct": round(e["root_err_share"], 1)} for w, e in T.items()} for ds, T in [("catsdogs", tcd), ("cifar10", tcf)]},
    "near_duplicates": json.loads((R / "catsdogs" / "near_duplicates.json").read_text()),
}
audit["summary"] = key
(ROOT / "provenance" / "label_noise_audit.json").write_text(json.dumps(audit, indent=1, default=float))
print(f"{len(audit['checks'])} checks passed; anomalies:"); [print("  ", a) for a in audit["anomalies"]]
