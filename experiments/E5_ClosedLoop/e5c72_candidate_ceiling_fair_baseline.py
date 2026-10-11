"""E5-C72: frozen causal-edge bottleneck audit and edge-budget-matched temporal baseline.
Run from D:\\DRTT\\E5_ClosedLoop. New synthetic seeds 7201..7205; no retuning.
"""
import csv, hashlib, json, pickle, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path.cwd(); SCRIPTS=ROOT/'src'/'causal'; sys.path.insert(0,str(SCRIPTS))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base
OUT=ROOT/'results'/'causal'/'e5c72_candidate_ceiling'
SEEDS=range(7201,7206); WINDOW=90

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):
 with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def load(p):
 if not p.is_file():raise FileNotFoundError(str(p))
 with p.open('rb') as f:return pickle.load(f) # Only trusted local model files.
def score(pred,truth):
 tp=len(pred&truth);fp=len(pred)-tp;fn=len(truth)-tp
 return {'predicted':len(pred),'tp':tp,'fp':fp,'fn':fn,'precision':tp/len(pred) if pred else 0.,'recall':tp/len(truth) if truth else 0.,'f1':2*tp/(len(pred)+len(truth)) if pred or truth else 0.}

def candidates(events,chosen,window):
 """For each selected child, enumerate eligible parents with E5-C58 exact rules.
 Return dict child -> nearest-first [(parent_id,gap),...].
 """
 ordered=sorted((e for e in events if e['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
 recent=deque();out={}
 for child in ordered:
  ct=int(child['timestamp'])
  while recent and int(recent[0]['timestamp'])<ct-window:recent.popleft()
  eligible=[]
  for p in reversed(recent):
   gap=ct-int(p['timestamp'])
   if gap<=0:continue
   if (p['destination_host']==child['source_host'] or p['source_host']==child['source_host'] or p['destination_host']==child['destination_host']):
    eligible.append((p['event_id'],gap))
  eligible.sort(key=lambda v:(v[1],v[0]))
  if eligible:out[child['event_id']]=eligible
  recent.append(child)
 return out

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
 'new':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
 'old_gate':ROOT/'results/causal/e5c31/no_parent_gate.pkl',
 'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
 'safety':ROOT/'results/causal/e5c64/constrained_gate_policy.json',
 'conservative':ROOT/'results/causal/e5c66a/summary.json'}
 for name,path in paths.items():
  if not path.is_file():raise SystemExit(f'Missing {name}: {path}')
 old,new,old_gate,gate=(load(paths[k]) for k in ('old','new','old_gate','gate'))
 if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1]:raise RuntimeError('Ranker schema mismatch')
 if gate['features']!=c58.GATE_FEATURES or old_gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Gate schema mismatch')
 if new.get('baseline_sha256')!=sha(paths['old']) or old_gate.get('e5c15_sha256')!=sha(paths['old']):raise RuntimeError('Ranker lineage mismatch')
 if gate.get('e5c56_sha256')!=sha(paths['new']) or gate.get('baseline_e5c15_sha256')!=sha(paths['old']):raise RuntimeError('Gate lineage mismatch')
 safety=json.loads(paths['safety'].read_text(encoding='utf-8'))
 summary=json.loads(paths['conservative'].read_text(encoding='utf-8'))
 if safety.get('experiment')!='E5-C64' or safety.get('model_sha256',{}).get('proposed_ranker')!=sha(paths['new']) or safety.get('model_sha256',{}).get('compatible_gate')!=sha(paths['gate']):raise RuntimeError('Safety provenance mismatch')
 if summary.get('experiment')!='E5-C66A':raise RuntimeError('Conservative provenance mismatch')
 best=summary['best_feasible_by_macro_f1']
 if abs(float(best['sparse_cutoff'])-.725)>1e-9 or abs(float(best['dense_cutoff'])-.8)>1e-9:raise RuntimeError('Conservative thresholds unexpected')
 if abs(float(gate['budget'])-1.5)>1e-9:raise RuntimeError('Alert budget unexpected')
 policies={'Global':(float(gate['cutoff']),float(gate['cutoff'])),
 'E5-C64':(float(safety['sparse_cutoff']),float(safety['dense_cutoff'])),
 'E5-C66A':(float(best['sparse_cutoff']),float(best['dense_cutoff']))}
 train=read(ROOT/'data/causal/e5c15_train/telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in read(ROOT/'data/causal/e5c15_train/ground_truth.csv')}
 feat=c38.extract(train,WINDOW)
 model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 model.fit(np.asarray([feat[r['event_id']] for r in train]),np.asarray([labels[r['event_id']] for r in train]))
 rows=[];traces=[]
 for seed in SEEDS:
  events,ground=gen.generate(seed,100,2400,10000)
  truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
  ids=[r['event_id'] for r in events];xf=c38.extract(events,WINDOW)
  probs=model.predict_proba(np.asarray([xf[i] for i in ids]))[:,1]
  n=min(len(events),round(sum(float(r['signal'])>=.65 for r in events)*1.5))
  chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
  alert_edges={(p,c) for p,c in truth if p in chosen and c in chosen}
  candidate_map=candidates(events,chosen,WINDOW)
  reachable={(p,c) for p,c in alert_edges if any(pid==p for pid,_ in candidate_map.get(c,[]))}
  learned=c58.top_rows(events,chosen,new,WINDOW)
  if len(learned)!=len(candidate_map) or {r['child'] for r in learned}!=set(candidate_map):raise RuntimeError('Candidate enumeration mismatch')
  ranked={(r['parent'],r['child']) for r in learned}
  gp=np.asarray(gate['model'].predict_proba(np.asarray([r['features'] for r in learned]))[:,1],dtype=float)
  counts=np.asarray([int(round(np.expm1(r['features'][4]))) for r in learned]);dense=counts>=16
  # Same-child baseline: choose most recent eligible parent for each learned candidate child.
  nearest={ (candidate_map[r['child']][0][0],r['child']) for r in learned }
  # Budget matched: compare equal numbers of predicted edges; temporal baseline ranks by smallest gap.
  near_ranked=sorted(((pairs[0][1],child,pairs[0][0]) for child,pairs in candidate_map.items()),key=lambda v:(v[0],v[1],v[2]))
  rows.append({'seed':seed,'policy':'CEILING','truth_edges':len(truth),'alerts':len(chosen),'candidate_children':len(candidate_map),
               'alert_reachable':len(alert_edges),'candidate_reachable':len(reachable),
               'ranker_correct':len(ranked&truth),'alert_recall_ceiling':len(alert_edges)/len(truth),
               'candidate_recall_ceiling':len(reachable)/len(truth),
               'ranker_top1_accuracy_on_reachable':len(ranked&truth)/len(reachable) if reachable else 0.,
               'ranker_recall_ceiling':len(ranked&truth)/len(truth),**{f'm_{k}':v for k,v in score(ranked,truth).items()}})
  for name,(s,d) in policies.items():
   keep=gp>=np.where(dense,d,s)
   pred={(r['parent'],r['child']) for r,k in zip(learned,keep) if k}
   budget=len(pred)
   temporal={(pid,child) for _,child,pid in near_ranked[:budget]}
   # Secondary, same-child control: gate-selected children with temporal parent instead of learned parent.
   same_child={(candidate_map[r['child']][0][0],r['child']) for r,k in zip(learned,keep) if k}
   for method,edges in [('Learned',pred),('TemporalHost_budget_matched',temporal),('TemporalHost_same_children',same_child)]:
    rows.append({'seed':seed,'policy':name,'method':method,'truth_edges':len(truth),'alerts':len(chosen),
                 'candidate_children':len(candidate_map),'alert_reachable':len(alert_edges),'candidate_reachable':len(reachable),
                 'ranker_correct':len(ranked&truth),'alert_recall_ceiling':len(alert_edges)/len(truth),
                 'candidate_recall_ceiling':len(reachable)/len(truth),
                 'ranker_top1_accuracy_on_reachable':len(ranked&truth)/len(reachable) if reachable else 0.,
                 'ranker_recall_ceiling':len(ranked&truth)/len(truth),**{f'm_{k}':v for k,v in score(edges,truth).items()}})
  if seed==7201:
   byid={r['event_id']:r for r in events}
   for r,prob in zip(learned,gp):
    pair=(r['parent'],r['child'])
    if pair in truth and len(traces)<8:
     child=r['child']; true_parent=next((p for p,c in truth if c==child),'')
     traces.append({'seed':seed,'child':child,'true_parent':true_parent,'selected_parent':r['parent'],
                    'gate_probability':float(prob),'candidate_count':len(candidate_map[child]),
                    'parent_signal':byid[r['parent']]['signal'],'child_signal':byid[child]['signal'],
                    'temporal_parent':candidate_map[child][0][0]})
  print(f'seed={seed} true={len(truth)} alert_ceiling={len(alert_edges)} candidate_ceiling={len(reachable)} ranked_correct={len(ranked&truth)}',flush=True)
 # Group summary with pooled edge counts and macro ceilings.
 groups=defaultdict(list)
 for r in rows:groups[(r['policy'],r.get('method','CEILING'))].append(r)
 summaries=[]
 for (policy,method),subset in groups.items():
  tp=sum(r['m_tp'] for r in subset);fp=sum(r['m_fp'] for r in subset);total=sum(r['truth_edges'] for r in subset)
  summaries.append({'policy':policy,'method':method,'seeds':len(subset),'total_truth':total,'tp':tp,'fp':fp,
    'pooled_f1':2*tp/(total+tp+fp) if total+tp+fp else 0,
    'mean_alert_ceiling':float(np.mean([r['alert_recall_ceiling'] for r in subset])),
    'mean_candidate_ceiling':float(np.mean([r['candidate_recall_ceiling'] for r in subset])),
    'mean_ranker_recall_ceiling':float(np.mean([r['ranker_recall_ceiling'] for r in subset])),
    'mean_ranker_top1_accuracy_on_reachable':float(np.mean([r['ranker_top1_accuracy_on_reachable'] for r in subset]))})
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'seed_metrics.csv',rows);dump(OUT/'summary.csv',summaries)
 if traces:dump(OUT/'worked_trace.csv',traces)
 (OUT/'audit.json').write_text(json.dumps({'experiment':'E5-C72','seeds':list(SEEDS),'window':WINDOW,'scenarios':'reference only','policies':policies,
  'model_sha256':{k:sha(paths[k]) for k in ('old','new','old_gate','gate')},
  'notes':['Frozen models; no retuning','Matched baseline has same edge count as learned policy','Same-child baseline holds selected children constant','Candidate ceiling counts true edges with both alerts and eligible host/time pair','Synthetic generator only; no external validation']},indent=2),encoding='utf-8')
 print('\nE5-C72 SUMMARY')
 for r in summaries:print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
