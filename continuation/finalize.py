"""Freeze checkpoint hashes before evaluating any test predictions."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from common import save,digest,probabilities,metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--results',required=True);p.add_argument('--plans',nargs='+',required=True);a=p.parse_args()
 out=Path(a.results);bank=Path(a.bank);jobs=[]
 for plan in a.plans:jobs.extend(json.loads(Path(plan).read_text())['jobs'])
 if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate job IDs between plans')
 checkpoints={}
 for j in jobs:
  d=out/j['id'];r=json.loads((d/'metrics.json').read_text())
  if r['job']!=j:raise ValueError('Result/plan mismatch')
  checkpoints[j['id']]=digest(d/'model.pkl')
  if checkpoints[j['id']]!=r.get('checkpoint_sha256'):raise ValueError('Checkpoint changed after fitting')
 lock=dict(plans={str(Path(p).name):digest(p) for p in a.plans},checkpoints=checkpoints,rule='All prespecified fits evaluated, not test-selected',bank=digest(bank/'metadata.json'))
 lp=out/'TEST_LOCK.json'
 if lp.exists() and json.loads(lp.read_text())!=lock:raise ValueError('Test lock differs: no retuning in this directory')
 save(lp,lock) # Happens BEFORE loading test features/labels.
 x=np.load(bank/'test_X.npy');y=np.load(bank/'test_y.npy')
 for j in jobs:
  d=out/j['id']
  if (d/'test_metrics.json').exists():
   previous=json.loads((d/'test_metrics.json').read_text())
   if digest(d/'test_predictions.npz')!=previous['predictions_sha256']:raise ValueError('Test prediction file changed')
   continue
  # Only locally generated trusted checkpoints, never arbitrary downloads.
  with open(d/'model.pkl','rb') as f:payload=pickle.load(f)
  pp=probabilities(payload['model'],((x-payload['mean'])/payload['std']).astype('float32'))
  np.savez_compressed(d/'test_predictions.npz',test=pp.astype('float32'))
  mm=metrics(pp,y);mm['predictions_sha256']=digest(d/'test_predictions.npz');save(d/'test_metrics.json',mm)
 print('Test evaluation complete. Do not tune against these scores.')
if __name__=='__main__':main()
