"""E5-C79: topology/temporal-aware alert selection; validation-only selection.
Run from D:\\DRTT\\E5_ClosedLoop. Frozen E5-C76 detectors and E5-C75 rankers/gates.
Validation 7901-7903, independent test 7904-7908. Synthetic-only.
No truth labels enter alert selection, graph construction, or inference.
"""
import csv, json, hashlib, heapq, random, sys, math
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76
OUT=ROOT/'results/causal/e5c79_structure_aware_alert_selection'
VAL=[7901,7902,7903];TEST=[7904,7905,7906,7907,7908]
W=90;ALERTS=[1300,1800];EDGE_BUDGET=400
POLICIES=[('ScoreOnly',0.,0.),('TopologyWeak',0.12,0.),('TopologyStrong',0.35,0.),('Temporal',0.,0.2),('TopologyTemporal',0.2,0.15)]

def write(p,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def variants(seed):
 ev,gt=gen.generate(seed,100,2400,10000)
 truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
 labels={r['event_id']:int(r['malicious']) for r in gt}
 benign=[r['signal'] for r in ev if labels[r['event_id']]==0]
 rng=random.Random(170000+seed)
 matched=[dict(r,signal=rng.choice(benign)) for r in ev]
 return truth,labels,{'original':ev,'matched_signal':matched}

def graph(events):
 """Host-compatible 1..90 time-neighborhood; symmetric for alert selection only."""
 n=len(events);order=sorted(range(n),key=lambda i:(int(events[i]['timestamp']),events[i]['event_id']))
 adj=[set() for _ in range(n)];near=[0]*n;active=deque()
 for i in order:
  a=events[i];t=int(a['timestamp'])
  while active and t-int(events[active[0]]['timestamp'])>W:active.popleft()
  for j in active:
   b=events[j];gap=t-int(b['timestamp'])
   if gap<=0:continue
   near[i]+=1;near[j]+=1
   if b['destination_host']==a['source_host'] or b['source_host']==a['source_host'] or b['destination_host']==a['destination_host']:
    adj[i].add(j);adj[j].add(i)
  active.append(i)
 return adj,near

def select(ids,probs,adj,near,k,alpha,beta):
 """Greedy score + bounded selected-neighbor bonus; deterministic, label-free.
 alpha rewards topology connections to already selected alerts; beta rewards temporal density.
 """
 n=len(ids);rank=sorted(range(n),key=lambda i:(-float(probs[i]),ids[i]))
 base=[0.]*n
 for pos,i in enumerate(rank):base[i]=1.-pos/max(n-1,1)
 if not alpha and not beta:return set(ids[i] for i in rank[:k])
 maxnear=max(max(near),1)
 time=[math.log1p(x)/math.log1p(maxnear) for x in near]
 count=[0]*n;selected=[False]*n;version=[0]*n
 def value(i):return base[i]+alpha*min(count[i],3)/3.+beta*time[i]
 heap=[(-value(i),ids[i],i,0) for i in range(n)];heapq.heapify(heap)
 chosen=[]
 while len(chosen)<min(k,n):
  neg,_,i,v=heapq.heappop(heap)
  if selected[i] or v!=version[i]:continue
  selected[i]=True;chosen.append(ids[i])
  if alpha:
   for j in adj[i]:
    if not selected[j] and count[j]<3:
     count[j]+=1;version[j]+=1
     heapq.heappush(heap,(-value(j),ids[j],j,version[j]))
 return set(chosen)

def evaluate(seed,truth,labels,variant,ev,detector,ranker,gate,policy,alerts,adj,near):
 ids=[r['event_id'] for r in ev];features=c38.extract(ev,W)
 probs=detector.predict_proba(np.asarray([features[i] for i in ids],dtype=float))[:,1]
 name,alpha,beta=policy
 chosen=select(ids,probs,adj,near,alerts,alpha,beta)
 reachable=sum(pa in chosen and ch in chosen for pa,ch in truth)
 rr=c58.top_rows(ev,chosen,ranker,W)
 if rr:
  p=gate['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1]
  pred=e75.top_budget(rr,p,EDGE_BUDGET)
 else:pred=set()
 tp=len(pred&truth)
 return dict(seed=seed,variant=variant,policy=name,alerts=alerts,edge_budget=EDGE_BUDGET,
  truth=len(truth),malicious_events=sum(labels.values()),malicious_alerts=sum(labels[i] for i in chosen),
  alert_reachable_edges=reachable,candidate_children=len(rr),predicted=len(pred),tp=tp,fp=len(pred)-tp,
  fn=len(truth)-tp,f1=e75.f1(tp,len(pred),len(truth)))

def aggregate(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['policy'],r['alerts'],r['variant'])].append(r)
 out=[]
 for (policy,alerts,variant),rr in groups.items():
  truth=sum(x['truth'] for x in rr);pred=sum(x['predicted'] for x in rr);tp=sum(x['tp'] for x in rr)
  out.append(dict(policy=policy,alerts=alerts,variant=variant,seeds=len(rr),truth=truth,predicted=pred,
   tp=tp,fp=pred-tp,fn=truth-tp,malicious_alerts=sum(x['malicious_alerts'] for x in rr),
   event_recall=sum(x['malicious_alerts'] for x in rr)/sum(x['malicious_events'] for x in rr),
   alert_reachable_edges=sum(x['alert_reachable_edges'] for x in rr),
   alert_edge_recall=sum(x['alert_reachable_edges'] for x in rr)/truth,
   pooled_f1=e75.f1(tp,pred,truth)))
 return out

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite {OUT}')
 paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
 'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
 'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
 'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
 'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}
 for k,p in paths.items():
  if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
 old=e75.load(paths['old']);frozen=e75.load(paths['frozen']);hard=e75.load(paths['hard']);gate=e75.load(paths['gate']);new_gate=e75.load(paths['new_gate'])
 if old['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1] or hard['features']!=base.FEATURES[:-1]:raise RuntimeError('Ranker schema mismatch')
 if gate['features']!=c58.GATE_FEATURES or new_gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Gate schema mismatch')
 detectors=e76.fit_models()
 # Fixed detector/ranker choice from E5-C77 validation, not selected on E5-C79 test.
 detector=detectors['OriginalOnly'];ranker=hard;gating=new_gate
 def run(seeds,policies,alert_values,split):
  rows=[]
  for seed in seeds:
   truth,labels,evmap=variants(seed)
   for variant,ev in evmap.items():
    adj,near=graph(ev)
    ids=[r['event_id'] for r in ev];features=c38.extract(ev,W)
    scores=detector.predict_proba(np.asarray([features[i] for i in ids],dtype=float))[:,1]
    for alerts in alert_values:
     for policy in policies:
      chosen=select(ids,scores,adj,near,alerts,policy[1],policy[2])
      rr=c58.top_rows(ev,chosen,ranker,W)
      if rr:
       gp=gating['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1]
       pred=e75.top_budget(rr,gp,EDGE_BUDGET)
      else:pred=set()
      tp=len(pred&truth)
      rows.append(dict(split=split,seed=seed,variant=variant,policy=policy[0],alerts=alerts,
       edge_budget=EDGE_BUDGET,truth=len(truth),malicious_events=sum(labels.values()),
       malicious_alerts=sum(labels[i] for i in chosen),alert_reachable_edges=sum(pa in chosen and ch in chosen for pa,ch in truth),
       candidate_children=len(rr),predicted=len(pred),tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,
       f1=e75.f1(tp,len(pred),len(truth))))
   print(f'{split} seed={seed} complete',flush=True)
  return rows
 validation=run(VAL,POLICIES,ALERTS,'validation');vsum=aggregate(validation)
 # Select structure-aware policy on mean original+matched F1; same budget for control.
 options=[]
 for policy,_,_ in POLICIES[1:]:
  for alerts in ALERTS:
   sub=[r for r in vsum if r['policy']==policy and r['alerts']==alerts]
   if len(sub)!=2:raise RuntimeError('Missing validation conditions')
   score=sum(r['pooled_f1'] for r in sub)/2
   options.append(dict(policy=policy,alerts=alerts,validation_mean_f1=score,
    validation_original_f1=next(r['pooled_f1'] for r in sub if r['variant']=='original'),
    validation_matched_f1=next(r['pooled_f1'] for r in sub if r['variant']=='matched_signal')))
 winner=sorted(options,key=lambda r:(-r['validation_mean_f1'],r['alerts'],r['policy']))[0]
 selected=next(p for p in POLICIES if p[0]==winner['policy'])
 print('Validation winner:',winner,flush=True)
 test=run(TEST,[POLICIES[0],selected],[winner['alerts']],'test');summary=aggregate(test)
 OUT.mkdir(parents=True,exist_ok=False)
 write(OUT/'validation_seed_metrics.csv',validation);write(OUT/'validation_summary.csv',vsum)
 write(OUT/'validation_policy_selection.csv',options);write(OUT/'selected_configuration.csv',[winner])
 write(OUT/'test_seed_metrics.csv',test);write(OUT/'test_summary.csv',summary)
 (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C79',train_seeds=e76.TRAIN,
  validation_seeds=VAL,test_seeds=TEST,policies=POLICIES,alert_grid=ALERTS,edge_budget=EDGE_BUDGET,
  selected=winner,detector='OriginalOnly',ranker='HardNegative',gate='E5-C75',
  input_sha256={k:sha(p) for k,p in paths.items()},
  limitations=['Synthetic-only','Matched-signal perturbation uses ground-truth benign labels to create evaluation distribution',
  'Validation selected among four structure-aware variants; test compares only winner with same-budget score-only control',
  'No new detector or gate training','Topology heuristic may favor dense benign neighborhoods',
  'No confidence intervals; planned for E5-C82']),indent=2),encoding='utf-8')
 print('\nE5-C79 TEST SUMMARY')
 for r in summary:print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
