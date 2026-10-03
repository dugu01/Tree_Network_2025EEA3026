"""Node pre-threshold Grad-CAM and occlusion on validation images only.

These explain the local node score, not the nondifferentiable full tree.
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets
from data_features import encoder,device_choice,RecordDataset,load_features,save_json


def explain_run(run_dir,device="auto",data_root="data",max_nodes=3):
    from run import read_model
    run_dir=Path(run_dir)
    checkpoint=read_model(run_dir/"tree"/"model.pkl")
    forest=checkpoint["model"]
    cache=Path(json.loads((run_dir/"cache_reference.json").read_text())["path"])
    arrays,meta,mean,std=load_features(cache)
    rows=json.loads((cache/"split_manifest.json").read_text())["val"]
    dev=device_choice(device);net,transform=encoder(dev)
    cifar_train=cifar_test=None
    if meta["dataset"]=="cifar10":
        cifar_train=datasets.CIFAR10(data_root,train=True,download=False)
        cifar_test=datasets.CIFAR10(data_root,train=False,download=False)
    dataset=RecordDataset(rows,transform,cifar_train,cifar_test)
    destination=run_dir/"explanations";destination.mkdir(exist_ok=True)
    candidates=[v for v in forest.nodes() if v.model is not None]
    # Prefer one root plus correction nodes to many roots.
    nodes=([v for v in candidates if v.depth==0][:1]+[v for v in candidates if v.depth>0])[:max_nodes]
    if len(nodes)<max_nodes:
        nodes+=( [v for v in candidates if v not in nodes][:max_nodes-len(nodes)] )
    mean_t=torch.tensor(mean,device=dev);std_t=torch.tensor(std,device=dev)
    records=[]
    for node in nodes:
        scores=node.base_score(arrays["val"][0]);top=np.argsort(scores)[-4:][::-1]
        index=int(top[0]);x=dataset[index][0].unsqueeze(0).to(dev).requires_grad_(True)
        local=node.model.to(dev).eval()
        fmap=net.features(x);fmap.retain_grad()
        z=net.avgpool(fmap).flatten(1)
        score=local((z-mean_t)/std_t).ravel()[0]
        score.backward()
        weights=fmap.grad.mean(dim=(2,3),keepdim=True)
        cam=torch.relu((weights*fmap).sum(1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy()
        cam=cam/max(float(cam.max()),1e-12)
        # 4x4 occlusion grid; zero in normalized pixel space is ImageNet mean.
        variants=[];size=x.shape[-1];edges=np.linspace(0,size,5,dtype=int)
        for r in range(4):
            for c in range(4):
                masked=x.detach().clone();masked[:,:,edges[r]:edges[r+1],edges[c]:edges[c+1]]=0
                variants.append(masked)
        with torch.no_grad():
            all_scores=[]
            for j in range(0,len(variants),4):
                zz=net(torch.cat(variants[j:j+4]))
                all_scores.append(local((zz-mean_t)/std_t).ravel().cpu().numpy())
        change=(float(score.detach().cpu())-np.concatenate(all_scores)).reshape(4,4)
        rgb=x.detach().cpu().numpy()[0].transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])
        rgb=np.clip(rgb,0,1)
        fig,axes=plt.subplots(1,3,figsize=(10,3.3))
        axes[0].imshow(rgb);axes[0].set_title("Validation image")
        axes[1].imshow(rgb);axes[1].imshow(cam,cmap="inferno",alpha=.5,vmin=0,vmax=1);axes[1].set_title("Node-score Grad-CAM")
        im=axes[2].imshow(change,cmap="coolwarm");axes[2].set_title("Score drop under occlusion")
        fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04)
        for ax in axes:ax.axis("off")
        fig.suptitle(f"{node.name}: local pre-threshold score {float(score.detach().cpu()):.3f}",fontsize=10)
        fig.tight_layout();name=node.name.replace(".","_")+".png"
        fig.savefig(destination/name,dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,len(top),figsize=(9,2.7))
        for ax,i in zip(np.atleast_1d(axes),top):
            ax.imshow(dataset.image(int(i)));ax.set_title(f"score {scores[i]:.2f}");ax.axis("off")
        fig.suptitle(f"Highest local activations among validation images: {node.name}",fontsize=10)
        fig.tight_layout();fig.savefig(destination/(node.name.replace(".","_")+"_examples.png"),dpi=150);plt.close(fig)
        records.append({"node":node.name,"validation_index":index,"sample_id":rows[index]["id"],
                        "score":float(score.detach().cpu()),"occlusion_score_drops":change.tolist(),
                        "target":"local pre-threshold node score; not full-tree prediction",
                        "figure":name})
        node.model.cpu()
    save_json(destination/"explanations.json",records)
    print(f"Saved validation-image explanations in {destination}",flush=True)
