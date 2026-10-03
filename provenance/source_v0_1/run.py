#!/usr/bin/env python3
"""Run doctor, selftest, pilot, main, sweep, finalize, report, or bundle.

Run from this project directory. Completed experiment configurations resume;
an interrupted individual model is retrained, not falsely marked complete.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import argparse
from pathlib import Path
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import pickle
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, log_loss
from scipy.special import expit,softmax
from threadpoolctl import threadpool_limits
from tree_model import TrainConfig,Forest,prune_forest
from data_features import prepare,load_features,save_json,digest_file,device_choice

ROOT=Path(__file__).resolve().parent
VERSION="0.1.0"


def source_digest():
    h=hashlib.sha256()
    for p in sorted(ROOT.glob("*.py")):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    names=["torch","torchvision","numpy","scipy","scikit-learn","matplotlib","Pillow"]
    versions={n:importlib.metadata.version(n) for n in names}
    return {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
            "versions":versions,"mps":torch.backends.mps.is_available(),"cuda":torch.cuda.is_available(),
            "source_sha256":source_digest(),"package_version":VERSION}


def save_model(path,model,mean,std,meta):
    path=Path(path);tmp=path.with_suffix(".tmp")
    with open(tmp,"wb") as f:pickle.dump({"model":model,"mean":mean,"std":std,"metadata":meta},f)
    tmp.replace(path)


def read_model(path):
    # Only load this package's own locally produced checkpoints.
    with open(path,"rb") as f:return pickle.load(f)


def evaluate(model,X,y,include_confusion=False):
    t=time.perf_counter();pred=model.predict(X);seconds=time.perf_counter()-t
    scores=model.scores(X)
    probabilities=np.column_stack([1-expit(scores[:,0]),expit(scores[:,0])]) if len(model.classes)==2 else softmax(scores,axis=1)
    result={"accuracy":float(np.mean(pred==y)),"errors":int(np.sum(pred!=y)),"n":len(y),
            "log_loss":float(log_loss(y,probabilities,labels=model.classes)),"seconds":seconds}
    if include_confusion:result["confusion_matrix"]=confusion_matrix(y,pred,labels=model.classes).tolist()
    # Multiclass softmax scores are descriptive, not calibrated probabilities.
    return result


def write_metrics(out,model,arrays,meta,config,kind,start,selection=None):
    metrics={"dataset":meta["dataset"],"profile":meta["profile"],"kind":kind,"config":config,
             "cache_manifest_sha256":meta["manifest_sha256"],"head_parameters":model.n_parameters(),
             "trunk_parameters":meta["trunk_parameters"],"nodes":len(model.nodes()),
             "train":evaluate(model,*arrays["train"]),"val":evaluate(model,*arrays["val"]),
             "training_seconds":time.perf_counter()-start,"test":None,"environment":environment()}
    metrics["total_parameters"]=metrics["head_parameters"]+metrics["trunk_parameters"]
    X=arrays["val"][0]
    normal=model.predict(X);fast,count=model.predict_certified(X)
    if not np.array_equal(normal,fast):raise AssertionError("Certified and full predictions disagree")
    full=len(X)*len(model.nodes())
    metrics["certified_inference"]={"exact_agreement":True,"node_example_evaluations":count,
                                    "full_node_example_evaluations":full,"fraction_saved":1-count/full}
    # Repeated warmed-up median timing; overhead can make skipping slower.
    durations={}
    for name,fn in [("full",model.predict),("certified",model.predict_certified)]:
        fn(X);times=[]
        for _ in range(3):
            t=time.perf_counter();fn(X);times.append(time.perf_counter()-t)
        durations[name]=float(np.median(times))
    metrics["certified_inference"]["median_seconds"]=durations
    if selection is not None:metrics["pruning_events"]=selection
    save_json(out/"node_statistics.json",[{**v.stats,"history":v.history} for v in model.nodes()])
    save_json(out/"metrics.json",metrics)
    return metrics


def experiment(cache,out,config,prune=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arrays,meta,mean,std=load_features(cache)
    cfg=vars(config).copy()
    request={"config":cfg,"cache_manifest_sha256":meta["manifest_sha256"],"source_sha256":source_digest()}
    request_path=out/"request.json"
    if request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError(f"Configuration/source changed in {out}. Use a different --tag; old results remain preserved.")
    save_json(request_path,request)
    save_json(out/"cache_reference.json",{"path":str(Path(cache).resolve())})
    X,y=arrays["train"]
    classes=np.unique(y)
    for kind,depth in [("root",0),("tree",config.depth)]:
        dest=out/kind;dest.mkdir(exist_ok=True)
        if (dest/"metrics.json").exists():
            print(f"Completed {dest}; reusing",flush=True);continue
        start=time.perf_counter();local=copy.copy(config);local.depth=depth
        print(f"Training {kind}: {meta['dataset']}, {len(y)} images, width={config.width}, depth={depth}",flush=True)
        model=Forest(local,classes).fit(X,y)
        save_model(dest/"model.pkl",model,mean,std,meta)
        write_metrics(dest,model,arrays,meta,vars(local),kind,start)
    if prune:
        dest=out/"pruned";dest.mkdir(exist_ok=True)
        if not (dest/"metrics.json").exists():
            start=time.perf_counter();model=read_model(out/"tree"/"model.pkl")["model"]
            small,events=prune_forest(model,X,y,*arrays["val"],max_train_drop=0.,max_val_drop=.005)
            save_model(dest/"model.pkl",small,mean,std,meta)
            write_metrics(dest,small,arrays,meta,cfg,"pruned",start,events)
    return out


def finalize(run_dir):
    run_dir=Path(run_dir)
    candidates=[]
    for p in sorted(run_dir.glob("*/metrics.json")):
        m=json.loads(p.read_text());candidates.append((m["val"]["accuracy"],-m["total_parameters"],p.parent,m))
    if not candidates:raise ValueError("No completed models in --run-dir")
    # Selection is frozen and saved BEFORE reading any test features.
    best=max(candidates,key=lambda v:(v[0],v[1]))
    lock=run_dir/"selection_locked.json"
    selected={"rule":"highest validation accuracy; then fewer total parameters",
              "selected_kind":best[2].name,"selection_metrics_sha256":digest_file(best[2]/"metrics.json")}
    if not lock.exists():save_json(lock,selected)
    else:
        existing=json.loads(lock.read_text())
        if existing["selected_kind"]!=selected["selected_kind"]:raise ValueError("Selection changed after finalization")
    cache=json.loads((run_dir/"cache_reference.json").read_text())["path"]
    arrays,meta,_,_=load_features(cache)
    for _,_,folder,m in candidates:
        if m["test"] is not None:continue
        model=read_model(folder/"model.pkl")["model"]
        m["test"]=evaluate(model,*arrays["test"],include_confusion=True)
        m["test_evaluation_note"]="Evaluation after within-run validation selection was frozen; do not tune against these test results."
        save_json(folder/"metrics.json",m)
    print(f"Final test evaluation saved. Selected model: {best[2].name}",flush=True)


def doctor():
    env=environment();print(json.dumps(env,indent=2))
    selected=device_choice();x=torch.ones(4,device=selected);v=(x*x).sum()
    assert v.item()==4
    print(f"Device check PASS: {selected}. MPS is optional; CPU fallback is supported.")
    if platform.system()=="Darwin" and platform.machine()!="arm64":
        print("WARNING: Intel/Rosetta Python detected on Mac; install an Apple-silicon Python environment for MPS.")
    save_json(ROOT/"validation"/"user_environment.json",env)


def run_images(args,profile):
    datasets=["cifar10","catsdogs"] if args.dataset=="both" else [args.dataset]
    for dataset in datasets:
        cache=Path(args.cache)/dataset/profile
        caps=None
        if args.train_cap:
            caps={"train":args.train_cap,"val":args.val_cap,"test":args.test_cap}
        prepare(dataset,cache,args.data_root,args.cats_root,profile,args.device,args.image_batch,args.split_seed,caps)
        cfg=TrainConfig(width=args.width,depth=args.depth,epochs=args.epochs or (30 if profile=="pilot" else 100),
                        seed=args.seed,device=device_choice(args.head_device),batch_size=args.head_batch,
                        fixed_epochs=args.fixed_epochs)
        out=Path(args.results)/dataset/profile/args.tag
        experiment(cache,out,cfg,prune=True)
        if not args.no_explain:
            from explain import explain_run
            explain_run(out,args.device,args.data_root,max_nodes=3)
    from reporting import build_report
    build_report(args.results,ROOT/"report",compile_pdf=args.compile_pdf)


def bundle(results,destination):
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        for base in [Path(results),ROOT/"report",ROOT/"validation"]:
            if not base.exists():continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".json",".csv",".png",".pdf",".tex",".txt",".md",".log"}:
                    arc=Path(base.name)/p.relative_to(base);z.write(p,arc)
    print(f"Share this results bundle: {dest.resolve()}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["doctor","selftest","pilot","main","sweep","finalize","report","bundle"])
    p.add_argument("--dataset",choices=["cifar10","catsdogs","both"],default="cifar10")
    p.add_argument("--data-root",default="data");p.add_argument("--cats-root",default="data")
    p.add_argument("--cache",default="cache");p.add_argument("--results",default="results")
    p.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    p.add_argument("--head-device",choices=["auto","cpu","mps","cuda"],default="cpu")
    p.add_argument("--image-batch",type=int,default=16);p.add_argument("--head-batch",type=int,default=512)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--epochs",type=int,default=0)
    p.add_argument("--width",type=int,default=64);p.add_argument("--depth",type=int,default=2)
    p.add_argument("--seed",type=int,default=17);p.add_argument("--split-seed",type=int,default=17)
    p.add_argument("--tag",default="baseline_v1");p.add_argument("--fixed-epochs",action="store_true")
    p.add_argument("--no-explain",action="store_true");p.add_argument("--compile-pdf",action="store_true")
    p.add_argument("--profile",choices=["pilot","main"],default="pilot")
    p.add_argument("--widths",default="8,32,128,512");p.add_argument("--seeds",default="17,29,43")
    p.add_argument("--run-dir");p.add_argument("--repo-url",default=None)
    p.add_argument("--destination",default="results_to_share.zip")
    p.add_argument("--train-cap",type=int,default=0);p.add_argument("--val-cap",type=int,default=500)
    p.add_argument("--test-cap",type=int,default=500)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.command=="doctor":doctor()
        elif args.command=="selftest":
            subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=ROOT,check=True)
        elif args.command in ["pilot","main"]:run_images(args,args.command)
        elif args.command=="sweep":
            if args.dataset=="both":raise ValueError("Sweep one dataset at a time")
            cache=Path(args.cache)/args.dataset/args.profile
            prepare(args.dataset,cache,args.data_root,args.cats_root,args.profile,args.device,args.image_batch,args.split_seed)
            for seed in map(int,args.seeds.split(",")):
                for width in map(int,args.widths.split(",")):
                    cfg=TrainConfig(width=width,depth=args.depth,epochs=args.epochs or 100,seed=seed,
                                    device=device_choice(args.head_device),batch_size=args.head_batch,fixed_epochs=True)
                    experiment(cache,Path(args.results)/args.dataset/args.profile/f"sweep_{args.tag}_w{width}_s{seed}",cfg,prune=False)
            from reporting import build_report
            build_report(args.results,ROOT/"report",compile_pdf=args.compile_pdf)
        elif args.command=="finalize":
            if not args.run_dir:raise ValueError("Supply --run-dir results/DATASET/PROFILE/TAG")
            finalize(args.run_dir)
            from reporting import build_report
            build_report(args.results,ROOT/"report",compile_pdf=args.compile_pdf)
        elif args.command=="report":
            from reporting import build_report
            build_report(args.results,ROOT/"report",args.repo_url,args.compile_pdf)
        elif args.command=="bundle":bundle(args.results,args.destination)


if __name__=="__main__":main()
