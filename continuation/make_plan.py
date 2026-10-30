"""Freeze an experiment grid before looking at test predictions."""
import argparse,itertools
from common import save

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',choices=['cifar10','catsdogs'],required=True)
 p.add_argument('--phase',choices=['pilot','capacity','ensemble','biasvariance','trajectory'],required=True)
 p.add_argument('--out',required=True);p.add_argument('--tag',default='');p.add_argument('--tree-widths',default='');p.add_argument('--width',type=int,default=32)
 p.add_argument('--widths',default='4,8,16,32,64,128,256');p.add_argument('--epochs',type=int,default=100)
 p.add_argument('--fraction',type=float,default=.25);p.add_argument('--seeds',default='17,29,43')
 a=p.parse_args();seeds=list(map(int,a.seeds.split(',')));widths=list(map(int,a.widths.split(',')));jobs=[]
 def add(**kw):
  j=dict(dataset=a.dataset,phase=a.phase,width=a.width,depth=2,epochs=a.epochs,fraction=a.fraction,data_seed=101,seed=17,noise=0.,noise_seed=1234,mode='augment',views=3,bank_id=0,ensemble_family=None)
  j.update(kw);j['id']=f'{a.tag+a.phase}_{len(jobs):04d}';jobs.append(j)
 if a.phase=='pilot':
  for mode,width in itertools.product(['none','repeat','augment'],[8,32]):add(mode=mode,width=width,epochs=10,fraction=.02)
 elif a.phase=='capacity':
  for noise,mode,seed,width in itertools.product([0.,.15],['none','augment'],seeds,widths):add(noise=noise,mode=mode,seed=seed,width=width,depth=0)
  # At least three actual-tree widths; denser tree sweeps can be generated explicitly.
  treewidths=list(map(int,a.tree_widths.split(','))) if a.tree_widths else sorted(set([widths[0],widths[len(widths)//2],widths[-1]]))
  for noise,mode,seed,width in itertools.product([0.,.15],['none','augment'],seeds,treewidths):add(noise=noise,mode=mode,seed=seed,width=width)
  for seed,width,depth in itertools.product(seeds,treewidths,[0,2]):add(mode='repeat',seed=seed,width=width,depth=depth)
 elif a.phase=='ensemble':
  for family in ['none','augment_shared','augment_diverse','persistent_diverse']:
   for m,seed in enumerate([17,29,43,59,71,89,101]):
    add(seed=seed,fraction=1.,mode='none' if family=='none' else ('persistent' if family.startswith('persistent') else 'augment'),bank_id=m%3 if family.endswith('diverse') else 0,ensemble_family=family)
 elif a.phase=='biasvariance':
  for fraction,ds,seed,epochs,mode in itertools.product([.1,.25,.5],[101,211,307],[17,29],[10,30,100],['none','augment']):add(fraction=fraction,data_seed=ds,seed=seed,epochs=epochs,mode=mode)
 elif a.phase=='trajectory':
  for seed,noise,mode in itertools.product(seeds[:2],[0.,.15],['none','augment']):
   add(seed=seed,noise=noise,mode=mode,depth=0,epochs=max(a.epochs,500),checkpoints=sorted(set([1,5,10,20,30,50,75,100,150,200,300,500,max(a.epochs,500)])))
 from pathlib import Path
 if Path(a.out).exists():raise FileExistsError('Plan exists: preserve it; choose a new filename')
 save(a.out,dict(version=1,dataset=a.dataset,phase=a.phase,jobs=jobs,test_policy='No test predictions until all jobs in this plan are complete and checkpoint hashes are locked.',epoch_definition='epochs per node; tree budget curves are not a single SGD trajectory',tie_rule='smallest class index',pretraining='Frozen ImageNet MobileNetV3-Small',notes='Sample-size study counts original images, not views. All classes use the same fixed splits.'))
 print(len(jobs),'jobs written to',a.out)
if __name__=='__main__':main()
