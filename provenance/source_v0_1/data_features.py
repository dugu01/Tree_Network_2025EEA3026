"""Official image data, deterministic splits, and cached frozen CNN features."""
from pathlib import Path
import hashlib
import json
import time
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
from PIL import Image
from sklearn.model_selection import train_test_split

CATS_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ENCODER = "mobilenet_v3_small_IMAGENET1K_V1_features_avgpool"


def digest_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2),encoding="utf8");temp.replace(path)


def device_choice(requested="auto"):
    if requested!="auto":
        if requested=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable; use --device cpu")
        if requested=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        return requested
    if torch.cuda.is_available():return "cuda"
    if torch.backends.mps.is_available():return "mps"
    return "cpu"


def encoder(device):
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net=models.mobilenet_v3_small(weights=weights).eval().to(device)
    net.classifier=torch.nn.Identity()
    for p in net.parameters():p.requires_grad_(False)
    return net,weights.transforms()


def cap_indices(idx,y,cap,seed):
    if not cap or len(idx)<=cap:return np.asarray(idx)
    keep,_=train_test_split(np.asarray(idx),train_size=cap,stratify=np.asarray(y)[idx],random_state=seed)
    return np.sort(keep)


def find_petimages(root):
    root=Path(root)
    for candidate in [root,root/"PetImages",root/"catsdogs"/"PetImages"]:
        if (candidate/"Cat").is_dir() and (candidate/"Dog").is_dir():return candidate
    return None


def cats_records(root,download=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    pets=find_petimages(root)
    if pets is None and download:
        archive=root/"kagglecatsanddogs_5340.zip"
        if not archive.exists():
            print("Downloading Microsoft Cats vs Dogs archive (~787 MB)...",flush=True)
            part=archive.with_suffix(".part")
            urllib.request.urlretrieve(CATS_URL,part);part.replace(archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target=(root/name).resolve()
                if not target.is_relative_to(root.resolve()):raise ValueError("Unsafe archive path")
            z.extractall(root)
        pets=find_petimages(root)
    if pets is None:raise FileNotFoundError("Expected a PetImages folder containing Cat and Dog. Use --cats-root PATH.")
    rows=[];excluded=[];seen={}
    for label,folder in enumerate(["Cat","Dog"]):
        for p in sorted((pets/folder).iterdir()):
            if p.suffix.lower() not in {".jpg",".jpeg",".png"}:continue
            try:
                with Image.open(p) as im:
                    im=im.convert("RGB");im.load()
                    # Pixel digest catches identical decoded images even if metadata differs.
                    h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                if h in seen:
                    if seen[h]!=label:raise ValueError("Conflicting labels for identical decoded images")
                    excluded.append({"path":str(p),"reason":"duplicate decoded image"});continue
                seen[h]=label
                rows.append({"path":str(p.resolve()),"label":label,"id":h})
            except ValueError as e:
                if "Conflicting labels" in str(e):raise
                excluded.append({"path":str(p),"reason":str(e)})
            except Exception as e:excluded.append({"path":str(p),"reason":str(e)})
    if len(rows)<20:raise ValueError("Too few readable Cats vs Dogs images")
    return rows,excluded


class RecordDataset(Dataset):
    def __init__(self,records,transform,cifar_train=None,cifar_test=None):
        self.records=records;self.transform=transform;self.train=cifar_train;self.test=cifar_test
    def __len__(self):return len(self.records)
    def image(self,i):
        r=self.records[i]
        if "path" in r:
            with Image.open(r["path"]) as im:return im.convert("RGB")
        data=self.train if r["source"]=="train" else self.test
        return Image.fromarray(data.data[r["index"]])
    def __getitem__(self,i):return self.transform(self.image(i)),int(self.records[i]["label"])


def prepare(dataset,cache_path,data_root,cats_root,profile,device="auto",batch_size=16,seed=17,
            caps=None,download=True):
    cache_path=Path(cache_path)
    caps=caps or ({"train":2000,"val":500,"test":500} if profile=="pilot" else {"train":0,"val":0,"test":0})
    expected={"dataset":dataset,"profile":profile,"split_seed":seed,"encoder":ENCODER,
              "caps":caps,"data_root":str(Path(data_root).resolve()),"cats_root":str(Path(cats_root).resolve())}
    if (cache_path/"metadata.json").exists():
        meta=json.loads((cache_path/"metadata.json").read_text())
        if meta["request"]!=expected:raise ValueError("Existing cache configuration differs. Choose a new --cache directory.")
        for split in ["train","val","test"]:
            for suffix in ["X","y"]:
                p=cache_path/f"{split}_{suffix}.npy"
                if not p.exists() or digest_file(p)!=meta["file_hashes"][p.name]:raise ValueError("Feature cache is incomplete or changed")
        print(f"Reusing verified feature cache: {cache_path}",flush=True);return meta
    cache_path.mkdir(parents=True,exist_ok=True)
    data_root=Path(data_root);data_root.mkdir(parents=True,exist_ok=True)
    tr=te=None;excluded=[]
    if dataset=="cifar10":
        tr=datasets.CIFAR10(str(data_root),train=True,download=download)
        te=datasets.CIFAR10(str(data_root),train=False,download=download)
        y=np.asarray(tr.targets);idx=np.arange(len(y))
        train_idx,val_idx=train_test_split(idx,test_size=.1,stratify=y,random_state=seed)
        records={}
        for split,indices,source,ds in [("train",train_idx,"train",tr),("val",val_idx,"train",tr),
                                        ("test",np.arange(len(te)),"test",te)]:
            indices=cap_indices(indices,np.asarray(ds.targets),caps[split],seed)
            records[split]=[{"source":source,"index":int(i),"label":int(ds.targets[i]),"id":f"cifar10:{source}:{i}"} for i in indices]
        classes=tr.classes
    elif dataset=="catsdogs":
        rows,excluded=cats_records(cats_root,download)
        y=np.array([r["label"] for r in rows]);idx=np.arange(len(y))
        train_idx,other=train_test_split(idx,test_size=.3,stratify=y,random_state=seed)
        val_idx,test_idx=train_test_split(other,test_size=.5,stratify=y[other],random_state=seed)
        records={split:[rows[int(i)] for i in cap_indices(indices,y,caps[split],seed)]
                 for split,indices in [("train",train_idx),("val",val_idx),("test",test_idx)]}
        classes=["cat","dog"]
    else:raise ValueError(dataset)
    ids=[set(r["id"] for r in records[s]) for s in ["train","val","test"]]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise AssertionError("Split overlap")
    save_json(cache_path/"split_manifest.json",records)
    save_json(cache_path/"excluded_images.json",excluded)
    actual_device=device_choice(device);net,transform=encoder(actual_device)
    start=time.perf_counter()
    hashes={}
    for split,rows in records.items():
        loader=DataLoader(RecordDataset(rows,transform,tr,te),batch_size=batch_size,shuffle=False,num_workers=0)
        features=[];labels=[]
        for b,(x,y) in enumerate(loader):
            with torch.inference_mode():z=net(x.to(actual_device)).cpu().numpy()
            if not np.isfinite(z).all():raise FloatingPointError("Nonfinite CNN features")
            features.append(z);labels.append(y.numpy())
            if b%50==0:print(f"  {dataset}/{split}: {min((b+1)*batch_size,len(rows))}/{len(rows)} images",flush=True)
        for suffix,value in [("X",np.concatenate(features).astype(np.float32)),("y",np.concatenate(labels))]:
            p=cache_path/f"{split}_{suffix}.npy";np.save(p,value);hashes[p.name]=digest_file(p)
    meta={"request":expected,"dataset":dataset,"profile":profile,"classes":classes,
          "counts":{s:len(v) for s,v in records.items()},"feature_dim":int(features[0].shape[1]),
          "trunk_parameters":sum(p.numel() for p in net.parameters()),"device":actual_device,
          "seconds":time.perf_counter()-start,"file_hashes":hashes,
          "manifest_sha256":digest_file(cache_path/"split_manifest.json"),"excluded_count":len(excluded),
          "pretraining":"ImageNet-1K supervised; external data used", "normalization":"official pretrained weights transforms"}
    save_json(cache_path/"metadata.json",meta)
    print(f"Cached features at {cache_path}; counts {meta['counts']}",flush=True)
    return meta


def load_features(cache):
    p=Path(cache);meta=json.loads((p/"metadata.json").read_text())
    arrays={s:(np.load(p/f"{s}_X.npy"),np.load(p/f"{s}_y.npy")) for s in ["train","val","test"]}
    # Fit standardization on training data only, then apply unchanged elsewhere.
    mean=arrays["train"][0].mean(0);std=arrays["train"][0].std(0);std=np.maximum(std,1e-5)
    arrays={s:((x-mean)/std,y) for s,(x,y) in arrays.items()}
    return arrays,meta,mean,std
