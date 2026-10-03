"""Regenerate report tables/plots from saved JSON; does not train or evaluate."""
from pathlib import Path
import json,csv,statistics as st,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report';FIG=OUT/'figures';TAB=OUT/'tables'
FIG.mkdir(exist_ok=True,parents=True);TAB.mkdir(exist_ok=True,parents=True)
metrics=[]
for p in sorted((ROOT/'results').rglob('metrics.json')):
 m=json.loads(p.read_text());m['run']=p.parent.parent.name;m['path']=str(p.relative_to(ROOT));metrics.append(m)
sweep=[m for m in metrics if m['run'].startswith('sweep_capacity_clean_v1_')]
def rows(ds,w,k,seeds=(17,29,43)):
 return sorted([m for m in sweep if m['dataset']==ds and m['config']['width']==w and m['kind']==k and m['config']['seed'] in seeds],key=lambda m:m['config']['seed'])
def pm(values,d=2):
 return f'{st.mean(values):.{d}f}'+(r' $\pm$ '+f'{st.stdev(values):.{d}f}' if len(values)>1 else '')
def span(values):
 return str(min(values)) if min(values)==max(values) else f'{min(values)}--{max(values)}'
labels={'root':'Root','root_refit':'Root + refit','tree':'Tree','pruned':'Pruned'}
flat=[]
for m in metrics:
 r={k:m[k] for k in ['dataset','run','kind','nodes','head_parameters','total_parameters']};r.update(width=m['config']['width'],seed=m['config']['seed'],fixed_epochs=m['config']['fixed_epochs'])
 for split in ['train','val','test']:
  for k in ['n','errors','accuracy','log_loss']:r[f'{split}_{k}']=m[split][k] if m.get(split) else ''
 flat.append(r)
with open(TAB/'all_metrics.csv','w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=flat[0]);writer.writeheader();writer.writerows(flat)
lines=[]
for ds,w in [('cifar10',16),('catsdogs',8),('catsdogs',16)]:
 for k in labels:
  a=rows(ds,w,k);lines.append(f"{'CIFAR-10' if ds=='cifar10' else 'Cats/Dogs'} & {w} & {labels[k]} & {span([m['train']['errors'] for m in a])} & {pm([100*m['val']['accuracy'] for m in a])} & {pm([100*m['test']['accuracy'] for m in a])} \\\\")
(TAB/'main_rows.tex').write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
lines=[]
for ds,w in [('cifar10',16),('catsdogs',8),('catsdogs',16)]:
 for seed in [17,29,43]:
  t=rows(ds,w,'tree',[seed])[0];p=rows(ds,w,'pruned',[seed])[0]
  lines.append(f"{'CIFAR-10' if ds=='cifar10' else 'Cats/Dogs'} & {w} & {seed} & {t['nodes']} $\\to$ {p['nodes']} & {t['head_parameters']:,} $\\to$ {p['head_parameters']:,} & {100*(1-p['head_parameters']/t['head_parameters']):.2f} & {100*(1-p['total_parameters']/t['total_parameters']):.2f} \\\\")
(TAB/'prune_rows.tex').write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
lines=[]
for ds,w in [('cifar10',16),('catsdogs',8),('catsdogs',16),('cifar10',64),('cifar10',256),('catsdogs',32)]:
 for seed in ([17,29,43] if (ds,w) in [('cifar10',16),('catsdogs',8),('catsdogs',16)] else [17]):
  for k in labels:
   m=rows(ds,w,k,[seed])[0]
   lines.append(f"{'C10' if ds=='cifar10' else 'C/D'} & {w} & {seed} & {labels[k]} & {m['head_parameters']:,} & {m['train']['errors']} & {m['val']['errors']} & {m['test']['errors']} & {100*m['test']['accuracy']:.2f} \\\\")
(TAB/'exact_rows.tex').write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'axes.labelsize':8,'legend.fontsize':8,'savefig.dpi':220})
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
summary=[]
for row,(ds,interp) in enumerate([('cifar10',16),('catsdogs',8)]):
 for kind,color in [('root','#365e87'),('tree','#b46529')]:
  widths=sorted({m['config']['width'] for m in sweep if m['dataset']==ds})
  for col,(split,metric,mult,title) in enumerate([('train','errors',1,'Training error (%)'),('val','errors',1,'Validation error (%)'),('val','log_loss',1,'Validation log loss')]):
   yy=[];sd=[]
   for w in widths:
    a=rows(ds,w,kind);v=[m[split]['errors']/m[split]['n']*100 if metric=='errors' else m[split][metric] for m in a]
    yy.append(st.mean(v));sd.append(st.stdev(v))
    summary.append(dict(dataset=ds,kind=kind,width=w,measure=title,mean=yy[-1],sample_sd=sd[-1]))
   ax=axes[row,col];ax.errorbar(widths,yy,yerr=sd,color=color,marker='o',markersize=3,capsize=2,label=labels[kind],linewidth=1.3)
   ax.set_xscale('log',base=2);ax.set_xticks(widths,labels=widths);ax.set_xlabel('Node width');ax.set_title(('CIFAR-10' if ds=='cifar10' else 'Cats vs Dogs')+' | '+title);ax.grid(alpha=.18)
   if col==0:ax.set_yscale('symlog',linthresh=.01);ax.set_ylabel('Error (%)')
   elif col==1:ax.set_ylabel('Error (%)')
   else:ax.set_ylabel('Log loss')
 for ax in axes[row]:ax.axvline(interp,color='#777777',linestyle=':',linewidth=1,zorder=0)
 axes[row,0].legend()
fig.savefig(FIG/'capacity_curves.png');plt.close(fig)
with open(TAB/'sweep_summary.csv','w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
for ax,ds in zip(axes,['cifar10','catsdogs']):
 for kind,color in [('root','#365e87'),('tree','#b46529')]:
  for seed in [17,29,43]:
   a=sorted([m for m in sweep if m['dataset']==ds and m['kind']==kind and m['config']['seed']==seed],key=lambda m:m['config']['width'])
   ax.plot([m['head_parameters'] for m in a],[100*(1-m['val']['accuracy']) for m in a],'-o',alpha=.65,color=color,markersize=3,label=labels[kind] if seed==17 else None)
 ax.set_xscale('log');ax.set_xlabel('Actual retained head parameters');ax.set_ylabel('Validation error (%)');ax.set_title('CIFAR-10' if ds=='cifar10' else 'Cats vs Dogs');ax.grid(alpha=.18);ax.legend()
fig.savefig(FIG/'parameter_curves.png');plt.close(fig)
for ds,w,names,prefix in [('cifar10',16,['class_2','class_2_A','class_2_B'],'cifar'),('catsdogs',8,['class_1','class_1_A','class_1_A_B'],'cats')]:
 for name in names:
  shutil.copy2(ROOT/'results'/ds/'main'/f'sweep_capacity_clean_v1_w{w}_s17'/'explanations'/f'{name}.png',FIG/f'{prefix}_{name}.png')
print('Generated tables and figures from',len(metrics),'model records; no measurements invented.')
