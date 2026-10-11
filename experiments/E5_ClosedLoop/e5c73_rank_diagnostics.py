"""E5-C73 frozen ranker top-k audit and competitive same-child baselines.
Run from D:\\DRTT\\E5_ClosedLoop. No model fitting or threshold tuning on test seeds.
"""
import csv, json, pickle, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base
OUT=ROOT/'results'/'causal'/'e5c73_rank_diagnostics'
SEEDS=range(7301,7306); W=90

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def metric(edges,truth):
 tp=len(edges&truth);fp=len(edges)-tp;fn=len(truth)-tp
 return dict(tp=tp,fp=fp,fn=fn,predicted=len(edges),f1=2*tp/(len(edges)+len(truth)) if len(edges)+len(truth) else 0)
def rank_candidates(events,chosen,ranker):
 selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
 output={}
 for j,child in enumerate(selected):
  ct=int(child['timestamp']);pairs=[]
  for p in reversed(selected[:j]):
   gap=ct-int(p['timestamp'])
   if gap>W:break
   if gap<=0:continue
   if p['destination_host']==child['source_host'] or p['source_host']==child['source_host'] or p['destination_host']==child['destination_host']:
    pairs.append((p,gap))
  pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
  if not pairs:continue
  x=[]
  for i,(p,gap) in enumerate(pairs):
   x.append([gap/90.,float(p['source_host']==child['source_host']),float(p['destination_host']==child['destination_host']),float(p['destination_host']==child['source_host']),float(p['source_host']==child['destination_host']),float(p['sensor']==child['sensor']),float(p['event_type']==child['event_type']),float(p['signal']),float(child['signal']),len(pairs)/50.,i/50.])
  xx=np.asarray(x);features=xx if len(ranker['features'])==11 else xx[:,:-1]
  scores=np.asarray(ranker['model'].decision_function(features)).reshape(-1)
  # Fixed, label-blind heuristic: prefer host handoff, then repeated source, then smaller gap.
  heur=[(3*int(p['destination_host']==child['source_host'])+int(p['source_host']==child['source_host'])+int(p['destination_host']==child['destination_host'])-gap/90.,-gap,p['event_id'],p['event_id']) for p,gap in pairs]
  learned=sorted(range(len(pairs)),key=lambda i:(-float(scores[i]),pairs[i][1],pairs[i][0]['event_id']))
  temporal=sorted(range(len(pairs)),key=lambda i:(pairs[i][1],pairs[i][0]['event_id']))
  host=sorted(range(len(pairs)),key=lambda i:(-heur[i][0],pairs[i][1],pairs[i][0]['event_id']))
  output[child['event_id']]={'learned':[pairs[i][0]['event_id'] for i in learned],
                             'temporal':[pairs[i][0]['event_id'] for i in temporal],
                             'host':[pairs[i][0]['event_id'] for i in host],
                             'candidates':len(pairs)}
 return output

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'old':ROOT/'results/causal/e5c15/reranker.pkl','new':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
        'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl','safety':ROOT/'results/causal/e5c64/constrained_gate_policy.json',
        'conservative':ROOT/'results/causal/e5c66a/summary.json'}
 for k,p in paths.items():
  if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
 with paths['old'].open('rb') as f:old=pickle.load(f)
 with paths['new'].open('rb') as f:new=pickle.load(f)
 with paths['gate'].open('rb') as f:gate=pickle.load(f)
 if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1] or gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Schema mismatch')
 if new.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['new']) or gate.get('baseline_e5c15_sha256')!=sha(paths['old']):raise RuntimeError('Model lineage mismatch')
 safety=json.loads(paths['safety'].read_text());cons=json.loads(paths['conservative'].read_text())['best_feasible_by_macro_f1']
 if safety.get('model_sha256',{}).get('proposed_ranker')!=sha(paths['new']) or safety.get('model_sha256',{}).get('compatible_gate')!=sha(paths['gate']):raise RuntimeError('Policy lineage mismatch')
 policies={'Global':(float(gate['cutoff']),float(gate['cutoff'])),'E5-C64':(float(safety['sparse_cutoff']),float(safety['dense_cutoff'])),'E5-C66A':(float(cons['sparse_cutoff']),float(cons['dense_cutoff']))}
 train=read(ROOT/'data/causal/e5c15_train/telemetry.csv');ty={r['event_id']:int(r['malicious']) for r in read(ROOT/'data/causal/e5c15_train/ground_truth.csv')}
 feats=c38.extract(train,W);ids=[r['event_id'] for r in train]
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([ty[i] for i in ids]))
 results=[];traces=[]
 for seed in SEEDS:
  events,truthrows=gen.generate(seed,100,2400,10000)
  truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
  trueparent={c:p for p,c in truth}
  ids=[r['event_id'] for r in events];feat=c38.extract(events,W)
  probabilities=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
  n=min(len(events),round(sum(float(r['signal'])>=.65 for r in events)*float(gate['budget'])))
  chosen={i for i,_ in sorted(zip(ids,probabilities),key=lambda z:(-float(z[1]),z[0]))[:n]}
  ranks=rank_candidates(events,chosen,new)
  top=c58.top_rows(events,chosen,new,W)
  if set(ranks)!={r['child'] for r in top}:raise RuntimeError('Candidate mismatch')
  if any(r['parent']!=ranks[r['child']]['learned'][0] for r in top):raise RuntimeError('Ranker mismatch')
  gp=gate['model'].predict_proba(np.asarray([r['features'] for r in top]))[:,1]
  reachable=[c for c,p in trueparent.items() if c in ranks and p in ranks[c]['learned']]
  diag={'seed':seed,'truth':len(truth),'selected_alerts':len(chosen),'candidate_children':len(ranks),'reachable':len(reachable)}
  for k in (1,3,5,10):
   diag[f'learned_top{k}']=sum(trueparent[c] in ranks[c]['learned'][:k] for c in reachable)
   diag[f'host_top{k}']=sum(trueparent[c] in ranks[c]['host'][:k] for c in reachable)
  diag['temporal_top1']=sum(trueparent[c]==ranks[c]['temporal'][0] for c in reachable)
  print(f"seed={seed} reachable={len(reachable)} learned@1={diag['learned_top1']} learned@3={diag['learned_top3']} host@1={diag['host_top1']}",flush=True)
  for policy,(s,d) in policies.items():
   selected=[]
   for r,prob in zip(top,gp):
    dense=int(round(np.expm1(r['features'][4])))>=16
    if prob>=(d if dense else s):selected.append((r,float(prob)))
   children=[r['child'] for r,_ in selected]
   methods={
    'Learned':{(r['parent'],r['child']) for r,_ in selected},
    'Temporal_same_children':{(ranks[c]['temporal'][0],c) for c in children},
    'HostHandoff_same_children':{(ranks[c]['host'][0],c) for c in children},
   }
   # Matched edge count, identical child set. This avoids the earlier zero-TP
   # artifact from choosing unrelated children by globally smallest time gaps.
   for name,edges in methods.items():results.append({'seed':seed,'policy':policy,'method':name,**diag,**metric(edges,truth)})
   if policy=='Global':
    for r,prob in selected:
     c=r['child'];p=trueparent.get(c)
     if p and p in ranks[c]['learned'] and len(traces)<30:
      traces.append({'seed':seed,'child':c,'truth_parent':p,'learned_parent':r['parent'],
       'temporal_parent':ranks[c]['temporal'][0],'host_parent':ranks[c]['host'][0],
       'true_parent_rank':ranks[c]['learned'].index(p)+1,'gate_probability':prob,
       'learned_correct':int(r['parent']==p),'candidate_count':ranks[c]['candidates']})
 # summarize by method; ceilings are per-seed and identical across methods
 groups=defaultdict(list)
 for r in results:groups[(r['policy'],r['method'])].append(r)
 summaries=[]
 for (policy,method),rr in groups.items():
  total=sum(r['truth'] for r in rr);tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr);reach=sum(r['reachable'] for r in rr)
  summaries.append({'policy':policy,'method':method,'seeds':len(rr),'truth':total,'reachable':reach,
   'tp':tp,'fp':fp,'pooled_f1':2*tp/(total+tp+fp) if total+tp+fp else 0,
   **{f'{key}_count':sum(r[key] for r in rr) for key in ('learned_top1','learned_top3','learned_top5','learned_top10','host_top1','host_top3','temporal_top1')},
   'conditional_top1_accuracy':sum(r['learned_top1'] for r in rr)/reach if reach else 0})
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'seed_metrics.csv',results);dump(OUT/'summary.csv',summaries);dump(OUT/'worked_trace.csv',traces)
 (OUT/'audit.json').write_text(json.dumps({'experiment':'E5-C73','seeds':list(SEEDS),'models_sha256':{k:sha(paths[k]) for k in ('old','new','gate')},
  'note':'Frozen synthetic-only audit; baselines share gate-selected child set and edge count. Host heuristic is predeclared, label-blind. No model tuning.'},indent=2))
 print('\nE5-C73 SUMMARY')
 for row in summaries:print(row)
 print('Saved:',OUT)
if __name__=='__main__':main()
