"""E5-C61: frozen density-stratified reconstruction failure audit of E5-C60.
No training, model selection, or changes to frozen predictions.
"""
import argparse,csv,hashlib,json,pickle,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCENARIOS={'reference':(100,2400,10000),'background_half':(100,2400,5000),
 'background_double':(100,2400,20000),'campaigns_half':(50,2400,10000),
 'campaigns_onehalf':(150,2400,10000),'duration_half':(100,1200,10000),
 'duration_double':(100,4800,10000)}
BUCKETS=((1,1,'1'),(2,3,'2-3'),(4,7,'4-7'),(8,15,'8-15'),(16,10**9,'16+'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rank_candidates(events,chosen,ranker,window=90):
 """Reproduce E5-C58 candidate generation, feature order, decision score and tie-break."""
 selected=sorted((e for e in events if e['event_id'] in chosen),key=lambda e:(int(e['timestamp']),e['event_id']))
 output={}
 for j,child in enumerate(selected):
  t=int(child['timestamp']);pairs=[]
  for parent in reversed(selected[:j]):
   gap=t-int(parent['timestamp'])
   if gap>window:break
   if gap<=0:continue
   if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
   pairs.append((parent,gap))
  pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
  if not pairs:continue
  vec=[]
  for i,(parent,gap) in enumerate(pairs):
   vec.append([gap/90.,float(parent['source_host']==child['source_host']),float(parent['destination_host']==child['destination_host']),float(parent['destination_host']==child['source_host']),float(parent['source_host']==child['destination_host']),float(parent['sensor']==child['sensor']),float(parent['event_type']==child['event_type']),float(parent['signal']),float(child['signal']),len(pairs)/50.,i/50.])
  x=np.asarray(vec,dtype=float)
  z=x if len(ranker['features'])==11 else x[:,:-1]
  scores=np.asarray(ranker['model'].decision_function(z),dtype=float).reshape(-1)
  order=sorted(range(len(pairs)),key=lambda i:(-float(scores[i]),pairs[i][1],pairs[i][0]['event_id']))
  output[child['event_id']]=[pairs[i][0]['event_id'] for i in order]
 return output
def buck(n):
 return next(name for lo,hi,name in BUCKETS if lo<=n<=hi)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for x in ('ranker','retrained','gate','compatible-gate'):ap.add_argument('--'+x,type=Path,required=True)
 ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
 ap.add_argument('--reference',type=Path,default=Path('results/causal/e5c60/per_seed.csv'))
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--seed-start',type=int,default=6000)
 ap.add_argument('--num-seeds',type=int,default=5)
 a=ap.parse_args()
 if a.output.exists():ap.error('Output exists; refusing overwrite')
 for p in (a.ranker,a.retrained,a.gate,a.compatible_gate,a.reference,a.train/'telemetry.csv',a.train/'ground_truth.csv'):
  if not p.is_file():ap.error(f'Missing input: {p}')
 sys.path.insert(0,str(a.scripts.resolve()))
 import e5c15_reranker as base
 import e5c38_end_to_end as c38
 import e5c5_overlapping_generator as gen
 import e5c58_compatible_gate_calibration as c58
 with a.ranker.open('rb') as f:old=pickle.load(f)
 with a.retrained.open('rb') as f:new=pickle.load(f)
 with a.gate.open('rb') as f:og=pickle.load(f)
 with a.compatible_gate.open('rb') as f:ng=pickle.load(f)
 if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1]:raise ValueError('Ranker feature mismatch')
 if og['features']!=c58.GATE_FEATURES or ng['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature mismatch')
 if new.get('baseline_sha256')!=sha(a.ranker) or og.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Model lineage mismatch')
 if ng.get('e5c56_sha256')!=sha(a.retrained) or ng.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('Gate lineage mismatch')
 if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Expected frozen 1.5x budget')
 refs={(r['scenario'],int(r['seed'])):r for r in read(a.reference)}
 expected={(s,seed) for s in SCENARIOS for seed in range(a.seed_start,a.seed_start+a.num_seeds)}
 if set(refs)!=expected:raise ValueError('Reference scenario/seed rows do not match requested audit')
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train]
 feat=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 detail=[];per=[];buckets=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  for seed in range(a.seed_start,a.seed_start+a.num_seeds):
   events,truthrows=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
   truth_parent={c:p for p,c in truth}
   eids=[r['event_id'] for r in events]
   x=c38.extract(events,90)
   prob=detector.predict_proba(np.asarray([x[i] for i in eids]))[:,1]
   n=min(len(events),int(round(sum(float(e['signal'])>=.65 for e in events)*1.5)))
   chosen={i for i,_ in sorted(zip(eids,prob),key=lambda z:(-float(z[1]),z[0]))[:n]}
   ranked_old=rank_candidates(events,chosen,old)
   ranked_new=rank_candidates(events,chosen,new)
   old_rows=c58.top_rows(events,chosen,old,90)
   new_rows=c58.top_rows(events,chosen,new,90)
   if set(ranked_old)!=set(ranked_new) or set(ranked_old)!={r['child'] for r in old_rows}:raise RuntimeError('Candidate child mismatch')
   if any(ranked_old[r['child']][0]!=r['parent'] for r in old_rows) or any(ranked_new[r['child']][0]!=r['parent'] for r in new_rows):raise RuntimeError('Top1 mismatch')
   old_probs=og['model'].predict_proba(np.asarray([r['features'] for r in old_rows]))[:,1]
   new_probs=ng['model'].predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1]
   old_keep={r['child']:float(p)>=float(og['cutoff']) for r,p in zip(old_rows,old_probs)}
   new_keep={r['child']:float(p)>=float(ng['cutoff']) for r,p in zip(new_rows,new_probs)}
   old_accept={(r['parent'],r['child']) for r in old_rows if old_keep[r['child']]}
   new_accept={(r['parent'],r['child']) for r in new_rows if new_keep[r['child']]}
   tp0=len(old_accept&truth);fp0=len(old_accept-truth)
   tp1=len(new_accept&truth);fp1=len(new_accept-truth)
   ref=refs[(scenario,seed)]
   for key,val in (('baseline_tp',tp0),('baseline_fp',fp0),('proposed_tp',tp1),('proposed_fp',fp1),('alerts',n),('truth_edges',len(truth))):
    if int(ref[key])!=val:raise RuntimeError(f'E5-C60 mismatch {scenario} {seed} {key}: {val} vs {ref[key]}')
   eligible_old=sum(p in ranked_old.get(c,[]) for p,c in truth)
   eligible_new=sum(p in ranked_new.get(c,[]) for p,c in truth)
   if eligible_old!=eligible_new:raise RuntimeError('Candidate availability mismatch')
   bo=sum(ranked_old[c][0]==p for p,c in truth if c in ranked_old)
   bn=sum(ranked_new[c][0]==p for p,c in truth if c in ranked_new)
   per.append({'scenario':scenario,'seed':seed,'alerts':n,'candidate_children':len(ranked_old),
    'mean_candidates_per_child':float(np.mean([len(z) for z in ranked_old.values()])),
    'eligible_truth_edges':eligible_old,'baseline_top1_correct':bo,'proposed_top1_correct':bn,
    'baseline_gate_true_accepted':tp0,'proposed_gate_true_accepted':tp1,
    'baseline_top1_true_rejected':bo-tp0,'proposed_top1_true_rejected':bn-tp1,
    'baseline_false_accepted':fp0,'proposed_false_accepted':fp1,
    'baseline_f1':float(ref['baseline_f1']),'proposed_f1':float(ref['proposed_f1'])})
   for child,parents in ranked_old.items():
    true=truth_parent.get(child)
    np_=ranked_new[child]
    d={'scenario':scenario,'seed':seed,'child':child,'candidate_count':len(parents),
       'density_bucket':buck(len(parents)),'has_true_parent':int(true in parents),
       'baseline_true_parent_rank':parents.index(true)+1 if true in parents else 0,
       'proposed_true_parent_rank':np_.index(true)+1 if true in np_ else 0,
       'baseline_top1_correct':int(parents[0]==true),
       'proposed_top1_correct':int(np_[0]==true),
       'baseline_gate_accept':int(old_keep[child]),'proposed_gate_accept':int(new_keep[child]),
       'baseline_correct_accepted':int((parents[0],child) in old_accept and parents[0]==true),
       'proposed_correct_accepted':int((np_[0],child) in new_accept and np_[0]==true)}
    detail.append(d)
   print(f'{scenario} seed={seed}: mean candidates={per[-1]["mean_candidates_per_child"]:.2f} '
         f'eligible={eligible_old} top1={bo}->{bn} accepted correct={tp0}->{tp1} '
         f'false accepted={fp0}->{fp1}',flush=True)
 for scenario in SCENARIOS:
  for lo,hi,name in BUCKETS:
   sub=[r for r in detail if r['scenario']==scenario and lo<=r['candidate_count']<=hi]
   eligible=[r for r in sub if r['has_true_parent']]
   buckets.append({'scenario':scenario,'bucket':name,'candidate_children':len(sub),
    'eligible_truth_edges':len(eligible),
    'baseline_top1_correct':sum(r['baseline_top1_correct'] for r in eligible),
    'proposed_top1_correct':sum(r['proposed_top1_correct'] for r in eligible),
    'baseline_correct_accepted':sum(r['baseline_correct_accepted'] for r in eligible),
    'proposed_correct_accepted':sum(r['proposed_correct_accepted'] for r in eligible),
    'baseline_false_accepted':sum(r['baseline_gate_accept']-r['baseline_correct_accepted'] for r in sub),
    'proposed_false_accepted':sum(r['proposed_gate_accept']-r['proposed_correct_accepted'] for r in sub)})
 audit={'experiment':'E5-C61','status':'FROZEN DIAGNOSTIC; no fitting or tuning',
 'seed_range':[a.seed_start,a.seed_start+a.num_seeds-1],
 'reference_e5c60_sha256':sha(a.reference),
 'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),
 'baseline_gate':sha(a.gate),'proposed_gate':sha(a.compatible_gate)},
 'limitations':['Uses same E5-C60 seeds and scenarios; diagnostic only, not fresh independent validation',
 'Candidate count measures parent ambiguity, not direct physical network density',
 'Ranking and gate rejection are sequential error sources; counts are descriptive, not causal intervention estimates',
 'Ground truth used only after frozen inference; no model selection']}
 a.output.mkdir(parents=True,exist_ok=False)
 write(a.output/'per_seed.csv',per)
 write(a.output/'candidate_detail.csv',detail)
 write(a.output/'density_bucket_summary.csv',buckets)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('\nE5-C61 | SCENARIO DIAGNOSTIC')
 for s in SCENARIOS:
  rs=[r for r in per if r['scenario']==s]
  print(f'{s}: candidates/child={np.mean([r["mean_candidates_per_child"] for r in rs]):.2f} '
   f'eligible={sum(r["eligible_truth_edges"] for r in rs)} '
   f'top1={sum(r["baseline_top1_correct"] for r in rs)}->{sum(r["proposed_top1_correct"] for r in rs)} '
   f'gate_rejected_correct={sum(r["baseline_top1_true_rejected"] for r in rs)}->{sum(r["proposed_top1_true_rejected"] for r in rs)} '
   f'TP={sum(r["baseline_gate_true_accepted"] for r in rs)}->{sum(r["proposed_gate_true_accepted"] for r in rs)}')
 print('Saved:',a.output.resolve())
if __name__=='__main__':main()
