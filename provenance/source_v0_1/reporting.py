"""Regenerate report inputs from saved metrics; never invent measurements."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tex(s):
    replace={"\\":r"\textbackslash{}","_":r"\_","%":r"\%","&":r"\&","#":r"\#","$":r"\$","{":r"\{","}":r"\}"}
    return "".join(replace.get(c,c) for c in str(s))


def build_report(results,report,repo_url=None,compile_pdf=False):
    results=Path(results);report=Path(report);generated=report/"generated";generated.mkdir(parents=True,exist_ok=True)
    settings_path=report/"settings.json"
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {"repo_url":""}
    if repo_url is not None:
        if repo_url and not repo_url.startswith("https://github.com/"):raise ValueError("Use an https://github.com/OWNER/REPO link")
        settings["repo_url"]=repo_url
    settings_path.write_text(json.dumps(settings,indent=2))
    rows=[]
    for p in sorted(results.rglob("metrics.json")) if results.exists() else []:
        m=json.loads(p.read_text())
        if m.get("dataset") not in {"cifar10","catsdogs"}:continue
        m["run"]=p.parent.parent.name;m["folder"]=str(p.parent);rows.append(m)
    url=settings["repo_url"]
    (generated/"repository.tex").write_text((r"\url{"+url+"}" if url else r"\textbf{Pending: insert the actual private GitHub repository URL.}")+"\n")
    flat=[]
    for m in rows:
        flat.append({"dataset":m["dataset"],"profile":m["profile"],"run":m["run"],"kind":m["kind"],
                     "seed":m["config"]["seed"],"width":m["config"]["width"],"nodes":m["nodes"],
                     "train_n":m["train"]["n"],"train_accuracy":m["train"]["accuracy"],
                     "val_accuracy":m["val"]["accuracy"],"test_accuracy":m["test"]["accuracy"] if m["test"] else "pending",
                     "head_parameters":m["head_parameters"],"total_parameters":m["total_parameters"],
                     "train_seconds":m["training_seconds"],"certified_fraction_saved":m["certified_inference"]["fraction_saved"]})
    fields=list(flat[0]) if flat else ["dataset","profile","run","kind","train_accuracy","val_accuracy","test_accuracy"]
    with open(generated/"summary.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    lines=[]
    if not rows:
        lines=[r"\textbf{No image-training results are available yet.} The mathematical formulation and implementation are prepared; all dataset performance, double-descent, receptive-field and compression conclusions remain pending."]
    else:
        lines+=[r"The following values were read from saved run metrics. Pilot subsets and main splits are reported separately. A dash means the held-out test set has not yet been evaluated. No missing result is estimated."]
        for dataset in ["cifar10","catsdogs"]:
            for profile in ["pilot","main"]:
                group=[m for m in rows if m["dataset"]==dataset and m["profile"]==profile]
                if not group:continue
                lines+=[r"\subsection*{"+tex(dataset+" / "+profile)+"}",r"\begin{longtable}{@{}llrrrrr@{}}",
                        r"\toprule Run & Model & $N_{tr}$ & Nodes & Train (\%) & Val (\%) & Test (\%)\\ \midrule\endhead"]
                for m in group:
                    test=f"{100*m['test']['accuracy']:.2f}" if m["test"] else "--"
                    label=m["run"] if len(m["run"])<=24 else m["run"][:21]+"..."
                    lines.append(f"{tex(label)} & {tex(m['kind'])} & {m['train']['n']} & {m['nodes']} & {100*m['train']['accuracy']:.2f} & {100*m['val']['accuracy']:.2f} & {test} \\\\")
                lines+=[r"\bottomrule\end{longtable}",r"\begin{longtable}{@{}llrrr@{}}\toprule Run & Model & Head parameters & Total parameters & Fit time (s)\\\midrule\endhead"]
                for m in group:
                    label=m["run"] if len(m["run"])<=24 else m["run"][:21]+"..."
                    lines.append(f"{tex(label)} & {tex(m['kind'])} & {m['head_parameters']:,} & {m['total_parameters']:,} & {m['training_seconds']:.1f} \\\\")
                lines+=[r"\bottomrule\end{longtable}"]
    (generated/"results.tex").write_text("\n".join(lines)+"\n")
    # Exact numerical observations, not speculative visual explanations.
    observations=[]
    for m in rows:
        if m["kind"]!="tree" or m["run"].startswith("sweep_"):continue
        root=next((r for r in rows if (r["dataset"],r["profile"],r["run"],r["kind"])==(m["dataset"],m["profile"],m["run"],"root")),None)
        if root:
            dt=100*(m["train"]["accuracy"]-root["train"]["accuracy"]);dv=100*(m["val"]["accuracy"]-root["val"]["accuracy"])
            observations.append(tex(f"{m['dataset']} / {m['profile']} / {m['run']}: tree minus root accuracy was {dt:+.2f} percentage points on training and {dv:+.2f} on validation. Training errors remaining: {m['train']['errors']}.")+r"\par")
        ci=m["certified_inference"];full=ci["median_seconds"]["full"];fast=ci["median_seconds"]["certified"]
        observations.append(tex(f"Certified inference matched all {m['val']['n']} validation predictions. Node-example evaluations decreased by {100*ci['fraction_saved']:.2f}%; median full/certified inference times were {full:.4f}/{fast:.4f} seconds.")+r"\par")
    if not observations:observations=[r"Measured interpretations are pending. Synthetic correctness checks are not image-classification results."]
    observations.append(r"A human review of the figures and cross-seed trends is still required. A training accuracy near 100\% does not by itself establish harmful overfitting or double descent.")
    (generated/"observations.tex").write_text("\n\n".join(observations))
    figure_lines=[]
    for dataset in ["cifar10","catsdogs"]:
        for profile in ["pilot","main"]:
            group=[m for m in rows if m["dataset"]==dataset and m["profile"]==profile and m["kind"]=="tree" and m["run"].startswith("sweep_")]
            # Do not pool different sweep tags/protocols into one curve.
            tags=sorted(set(m["run"].rsplit("_w",1)[0] for m in group))
            for tag in tags:
                selected=[m for m in group if m["run"].rsplit("_w",1)[0]==tag]
                widths=sorted(set(m["config"]["width"] for m in selected))
                if len(widths)<2:continue
                fig,axes=plt.subplots(1,2,figsize=(10,3.5))
                for split,label in [("train","Training"),("val","Validation")]:
                    means=[];sds=[];xs=[]
                    for w in widths:
                        ms=[m for m in selected if m["config"]["width"]==w]
                        errs=[1-m[split]["accuracy"] for m in ms]
                        means.append(np.mean(errs));sds.append(np.std(errs,ddof=1) if len(errs)>1 else 0)
                        xs.append(np.mean([m["total_parameters"] for m in ms]))
                    axes[0].errorbar(widths,means,yerr=sds,marker="o",label=label,capsize=3)
                    axes[1].plot(xs,means,"o-",label=label)
                axes[0].set_xscale("log",base=2);axes[0].set_xlabel("MLP hidden width")
                axes[1].set_xscale("log");axes[1].set_xlabel("Mean total parameters (adaptive topology)")
                for ax in axes:ax.set_ylabel("Classification error");ax.grid(alpha=.2);ax.legend()
                fig.suptitle(f"{dataset} / {profile}: capacity sweep; bars = seed SD",fontsize=10)
                fig.tight_layout();name=f"capacity_{dataset}_{profile}_{tag}.png";fig.savefig(generated/name,dpi=160);plt.close(fig)
                figure_lines+=[r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{generated/"+name+r"}\caption{Capacity sweep. Maximum depth is controlled; actual grown topology and parameter count can change with width and seed. A double-descent conclusion requires inspection of interpolation and repeated-seed trends.}\end{figure}"]
    for dataset in ["cifar10","catsdogs"]:
        paths=sorted(results.glob(f"{dataset}/*/*/explanations/*_examples.png")) if results.exists() else []
        maps=sorted(p for p in results.glob(f"{dataset}/*/*/explanations/*.png") if not p.name.endswith("_examples.png")) if results.exists() else []
        for j,p in enumerate((maps[:1]+paths[:1])):
            name=f"node_{dataset}_{j}.png";shutil.copy2(p,generated/name)
            caption=tex(f"{dataset}: {p.parents[2].name} / {p.parents[1].name}. Validation images; local node score explanation, not an architectural receptive-field boundary.")
            figure_lines+=[r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{generated/"+name+r"}\caption{"+caption+r"}\end{figure}"]
    if not figure_lines:figure_lines=[r"Figures will be inserted after the corresponding image experiments complete."]
    (generated/"figures.tex").write_text("\n".join(figure_lines))
    checks=[]
    for dataset in ["cifar10","catsdogs"]:
        main=[m for m in rows if m["dataset"]==dataset and m["profile"]=="main" and m["kind"]=="tree"]
        checks.append(tex(f"{dataset}: "+("main tree results available" if main else "main tree results PENDING"))+r"\par")
        if main:checks.append(tex(f"Best recorded main training accuracy: {100*max(m['train']['accuracy'] for m in main):.3f}%. Report exact residual error count; target is not assumed achieved.")+r"\par")
    checks+=[r"Private repository URL: "+("provided (access not verified)." if url else "PENDING."),
             r"Final interpretations, repository access for the evaluator, and report review: PENDING."]
    (generated/"status.tex").write_text("\n\n".join(checks))
    if compile_pdf:
        if not shutil.which("pdflatex"):
            print("LaTeX source updated. pdflatex is unavailable; upload report/ to Overleaf or install a TeX distribution.")
        else:
            log=[]
            for _ in range(2):
                result=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","-jobname=Report_Draft","main.tex"],cwd=report,capture_output=True,text=True)
                log.append(result.stdout)
                if result.returncode:
                    (report/"build.log").write_text("\n".join(log));raise RuntimeError("LaTeX build failed; inspect report/build.log")
            (report/"build.log").write_text("\n".join(log))
    print(f"Updated report inputs and summary CSV: {report.resolve()}",flush=True)
