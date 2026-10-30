import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
name=Path('Continuation_'+a.dataset+'_evidence.zip')
with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
 for base in [Path('runs')/a.dataset,Path('analysis')/a.dataset,Path('plans')/a.dataset]:
  for f in base.rglob('*'):
   if f.is_file() and f.suffix in {'.json','.csv','.png','.npz','.log'}:z.write(f,f)
with zipfile.ZipFile(name) as z:assert z.testzip() is None
print(name.resolve())
