"""Augment RGB images BEFORE the frozen encoder; reuse verified original splits."""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from common import digest,save

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base-cache',required=True);p.add_argument('--out',required=True)
 p.add_argument('--data-root',default='data');p.add_argument('--cats-root');p.add_argument('--device',default='auto')
 p.add_argument('--aug-views',type=int,default=9);p.add_argument('--probe-views',type=int,default=2)
 p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=20261020)
 a=p.parse_args()
 import torch
 from torchvision import datasets,transforms as T
 from PIL import Image
 sys.path.insert(0,str(Path(__file__).parent/'baseline'))
 from data_features import encoder,device_choice
 b=Path(a.base_cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((b/'metadata.json').read_text());records=json.loads((b/'split_manifest.json').read_text())
 if digest(b/'split_manifest.json')!=meta['manifest_sha256']:raise ValueError('Original split manifest changed')
 for name,h in meta['file_hashes'].items():
  if digest(b/name)!=h:raise ValueError('Original feature hash mismatch: '+name)
 idsets=[{r['id'] for r in records[s]} for s in ('train','val','test')]
 assert not any(idsets[i]&idsets[j] for i in range(3) for j in range(i))
 request=dict(dataset=meta['dataset'],manifest=meta['manifest_sha256'],seed=a.seed,aug_views=a.aug_views,probe_views=a.probe_views,policy='mild_geometry_colour_v1',encoder=meta['request']['encoder'])
 if (out/'request.json').exists() and json.loads((out/'request.json').read_text())!=request:raise ValueError('Bank configuration changed: use new output directory')
 save(out/'request.json',request)
 for split in ('train','val','test'):
  for suf in ('X','y'):
   src=b/f'{split}_{suf}.npy';dest=out/src.name
   if not dest.exists():shutil.copy2(src,dest)
   if digest(dest)!=meta['file_hashes'][src.name]:raise ValueError('Copied cache mismatch')
 save(out/'ids.json',{s:[r['id'] for r in records[s]] for s in records})
 net,official=encoder(device_choice(a.device));device=next(net.parameters()).device
 cifar=datasets.CIFAR10(a.data_root,train=True,download=True) if meta['dataset']=='cifar10' else None
 if cifar is not None:
  augment=T.Compose([T.RandomCrop(32,padding=4,padding_mode='reflect'),T.RandomHorizontalFlip(),T.ColorJitter(.1,.1,.1,.02),official])
 else:
  augment=T.Compose([T.RandomResizedCrop(224,scale=(.8,1.),ratio=(.9,1.1)),T.RandomHorizontalFlip(),T.ColorJitter(.1,.1,.1,.02),official])
 def image(row):
  if cifar is not None:return Image.fromarray(cifar.data[row['index']])
  path=Path(row['path'])
  if a.cats_root:
   root=Path(a.cats_root);root=root/'PetImages' if (root/'PetImages').exists() else root
   path=root/path.parent.name/path.name
  with Image.open(path) as im:
   im=im.convert('RGB');im.load()
   import hashlib
   if hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()!=row['id']:raise ValueError('Image content differs: '+str(path))
   return im
 # Audit a reproducible sample of actual transformed images before long training.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 chosen=[]
 for label in sorted({r['label'] for r in records['train']}):
  chosen.extend([i for i,r in enumerate(records['train']) if r['label']==label][:2])
 chosen=chosen[:8]
 fig,axs=plt.subplots(len(chosen),4,figsize=(8,2*len(chosen)),squeeze=False)
 for row,i in enumerate(chosen):
  for col in range(4):
   torch.manual_seed(a.seed+(col-1)*1000000+i if col else a.seed+i)
   x=official(image(records['train'][i])) if col==0 else augment(image(records['train'][i]))
   rgb=(x.permute(1,2,0).numpy()*np.array(official.std)+np.array(official.mean)).clip(0,1)
   axs[row,col].imshow(rgb);axs[row,col].axis('off')
   axs[row,col].set_title(str(records['train'][i]['label'])+(' original' if col==0 else ' view '+str(col)))
 fig.tight_layout();fig.savefig(out/'augmentation_audit.png',dpi=140);plt.close(fig)
 hashes={}
 for kind,num in [('view',a.aug_views),('probe',a.probe_views)]:
  for v in range(num):
   dest=out/f'{kind}_{v}.npy';receipt=dest.with_suffix('.json')
   if dest.exists() and receipt.exists() and digest(dest)==json.loads(receipt.read_text())['sha256']:
    hashes[dest.name]=digest(dest);continue
   temporary=dest.with_suffix('.partial.npy');n=len(records['train']);d=meta['feature_dim']
   mm=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=(n,d))
   for start in range(0,n,a.batch):
    images=[]
    for i in range(start,min(start+a.batch,n)):
     torch.manual_seed(a.seed+(100000000 if kind=='probe' else 0)+v*1000000+i)
     images.append(augment(image(records['train'][i])))
    with torch.inference_mode():z=net(torch.stack(images).to(device)).cpu().numpy()
    if not np.isfinite(z).all():raise ValueError('Nonfinite features')
    mm[start:start+len(z)]=z
    if start%(a.batch*100)==0:print(kind,v,start,n,flush=True)
   mm.flush();del mm;temporary.replace(dest);hashes[dest.name]=digest(dest);save(receipt,{'sha256':hashes[dest.name]})
 allhash={f.name:digest(f) for f in out.glob('*.npy')}
 save(out/'metadata.json',dict(request=request,classes=meta['classes'],trunk_parameters=meta['trunk_parameters'],counts=meta['counts'],hashes=allhash,device=str(device),torch=torch.__version__,finite_bank=True))
 print('Bank complete:',out,flush=True)
if __name__=='__main__':main()
