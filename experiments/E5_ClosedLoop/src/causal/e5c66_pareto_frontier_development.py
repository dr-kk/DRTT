"""E5-C66: development-only F1/false-attribution/recall Pareto frontier.

Explore on synthetic seeds 6600-6604; check on 6605-6609. No reuse of E5-C63
independent seeds, and no new independent claim. Frozen models and detector fit.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCENARIOS={
 'reference':(100,2400,10000),
 'background_half':(100,2400,5000),
 'background_double':(100,2400,20000),
 'campaigns_half':(50,2400,10000),
 'campaigns_onehalf':(150,2400,10000),
 'duration_half':(100,1200,10000),
 'duration_double':(100,4800,10000),
}
FIT=range(6600,6605)
CHECK=range(6605,6610)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(tp,fp,truth_n):
 fn=truth_n-tp
 den=2*tp+fp+fn
 return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,
         'recall':tp/truth_n if truth_n else 0.,'f1':2*tp/den if den else 0.}

def aggregate(records,cs,cd):
 out={}
 for scenario in SCENARIOS:
  tp=fp=truth_n=0
  for r in records:
   if r['scenario']!=scenario:continue
   keep=r['probs']>=np.where(r['dense'],cd,cs)
   tp+=int(np.count_nonzero(keep&r['correct']))
   fp+=int(np.count_nonzero(keep&~r['correct']))
   truth_n+=r['truth_n']
  out[scenario]=metrics(tp,fp,truth_n)
 return out

def objective(stats):
 return float(np.mean([v['f1'] for v in stats.values()]))

def feasible(stats,global_stats,max_fp_ratio,reference_tolerance):
 for scenario in SCENARIOS:
  if stats[scenario]['fp']>max_fp_ratio*global_stats[scenario]['fp']+1e-9:
   return False
 if stats['reference']['f1']+1e-12 < global_stats['reference']['f1']-reference_tolerance:
  return False
 return True

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('ranker','retrained','gate','compatible-gate'):
  p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 p.add_argument('--scripts',type=Path,default=Path('src/causal'))
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-fp-ratio',type=float,default=1.25)
 p.add_argument('--reference-f1-tolerance',type=float,default=.005)
 p.add_argument('--safety-policy',type=Path,required=True)
 p.add_argument('--density-policy',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Output exists: refusing overwrite')
 if a.max_fp_ratio<1:p.error('max-fp-ratio must be >=1; global fallback must remain feasible')
 if a.reference_f1_tolerance<0:p.error('reference-f1-tolerance must be nonnegative')
 for path in (a.safety_policy,a.density_policy):
  if not path.is_file():p.error(f'Missing policy: {path}')
 for path in (a.ranker,a.retrained,a.gate,a.compatible_gate,a.train/'telemetry.csv',a.train/'ground_truth.csv'):
  if not path.is_file():p.error(f'Missing input: {path}')
 sys.path.insert(0,str(a.scripts.resolve()))
 import e5c15_reranker as base
 import e5c38_end_to_end as c38
 import e5c5_overlapping_generator as gen
 import e5c58_compatible_gate_calibration as c58
 with a.ranker.open('rb') as f:old=pickle.load(f)
 with a.retrained.open('rb') as f:new=pickle.load(f)
 with a.gate.open('rb') as f:og=pickle.load(f)
 with a.compatible_gate.open('rb') as f:ng=pickle.load(f)
 if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1]:raise ValueError('Ranker feature schema mismatch')
 if og['features']!=c58.GATE_FEATURES or ng['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature schema mismatch')
 if new.get('baseline_sha256')!=sha(a.ranker) or og.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Model lineage mismatch')
 if ng.get('e5c56_sha256')!=sha(a.retrained) or ng.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('Gate lineage mismatch')
 if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Expected frozen budget 1.5')
 frozen=float(ng['cutoff'])
 if abs(frozen-.7437635536351849)>1e-6:raise ValueError('Unexpected global cutoff')
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train]
 features=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([features[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 records=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  for seed in list(FIT)+list(CHECK):
   events,ground=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
   event_ids=[r['event_id'] for r in events]
   event_features=c38.extract(events,90)
   probs=detector.predict_proba(np.asarray([event_features[i] for i in event_ids]))[:,1]
   n=min(len(events),int(round(sum(float(r['signal'])>=.65 for r in events)*1.5)))
   chosen={i for i,_ in sorted(zip(event_ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
   rows=c58.top_rows(events,chosen,new,90)
   if not rows:raise RuntimeError('No candidate children')
   gate_probs=np.asarray(ng['model'].predict_proba(np.asarray([r['features'] for r in rows]))[:,1],dtype=float)
   cnt=np.asarray([int(round(np.expm1(r['features'][4]))) for r in rows],dtype=int)
   if np.any(cnt<1):raise RuntimeError('Invalid candidate count')
   correct=np.asarray([(r['parent'],r['child']) in truth for r in rows],dtype=bool)
   records.append({'scenario':scenario,'seed':seed,'truth_n':len(truth),'probs':gate_probs,'correct':correct,'dense':cnt>=16})
   print(f'{scenario} seed={seed}: truth={len(truth)} alerts={n} candidates={len(rows)} dense={int(np.sum(cnt>=16))}',flush=True)
 fit=[r for r in records if r['seed'] in FIT]
 check=[r for r in records if r['seed'] in CHECK]
 global_fit=aggregate(fit,frozen,frozen)
 policies={}
 for name,path in [('E5-C62',a.density_policy),('E5-C64',a.safety_policy)]:
  obj=json.loads(path.read_text(encoding='utf-8'))
  if obj.get('experiment')!=name:raise ValueError(f'Wrong policy experiment: {path}')
  if obj.get('model_sha256',{}).get('proposed_ranker')!=sha(a.retrained) or obj.get('model_sha256',{}).get('compatible_gate')!=sha(a.compatible_gate):
   raise ValueError(f'Policy model hash mismatch: {path}')
  if abs(float(obj['original_global_cutoff'])-frozen)>1e-9 or int(obj['density_boundary'])!=16:
   raise ValueError(f'Policy configuration mismatch: {path}')
  policies[name]=(float(obj['sparse_cutoff']),float(obj['dense_cutoff']))
 policies['Global']=(frozen,frozen)
 # Predeclared grid. No independent-test seeds are used for policy selection.
 grid=sorted(set([round(float(x),3) for x in np.arange(.30,.951,.025)]+[frozen]+[v for pair in policies.values() for v in pair]))
 curve=[]
 for cs in grid:
  for cd in grid:
   stats=aggregate(fit,cs,cd)
   row={'sparse_cutoff':cs,'dense_cutoff':cd,
        'macro_f1':objective(stats),
        'macro_recall':float(np.mean([v['recall'] for v in stats.values()])),
        'total_fp':sum(v['fp'] for v in stats.values()),
        'total_tp':sum(v['tp'] for v in stats.values()),
        'feasible':int(feasible(stats,global_fit,a.max_fp_ratio,a.reference_f1_tolerance)),
        'reference_f1':stats['reference']['f1'],
        'max_fp_ratio':max((stats[s]['fp']/global_fit[s]['fp'] if global_fit[s]['fp'] else float('inf')) for s in SCENARIOS)}
   curve.append(row)
 # Pareto dominance: F1 >=, recall >=, FP <= with at least one strict.
 # Vectorized across grid; no statistical claim from development Pareto positions.
 f1=np.asarray([r['macro_f1'] for r in curve]); recall=np.asarray([r['macro_recall'] for r in curve]); fp=np.asarray([r['total_fp'] for r in curve])
 frontier=[]
 for i,r in enumerate(curve):
  dominates=(f1>=f1[i]-1e-12)&(recall>=recall[i]-1e-12)&(fp<=fp[i])&((f1>f1[i]+1e-12)|(recall>recall[i]+1e-12)|(fp<fp[i]))
  r['pareto']=int(not np.any(dominates))
  if r['pareto']:frontier.append(r)
 frontier.sort(key=lambda r:(r['total_fp'],-r['macro_f1']))
 # Show how previously frozen policies behave on *new development* seeds.
 comparisons=[]
 for split,subset in [('exploration',fit),('development_check',check)]:
  global_stats=aggregate(subset,frozen,frozen)
  for name,(cs,cd) in policies.items():
   stats=aggregate(subset,cs,cd)
   row={'split':split,'policy':name,'sparse_cutoff':cs,'dense_cutoff':cd,
        'macro_f1':objective(stats),
        'macro_recall':float(np.mean([v['recall'] for v in stats.values()])),
        'total_fp':sum(v['fp'] for v in stats.values()),
        'total_tp':sum(v['tp'] for v in stats.values()),
        'constraints_met':int(feasible(stats,global_stats,a.max_fp_ratio,a.reference_f1_tolerance))}
   comparisons.append(row)
   print(f'{split} {name}: macro_F1={row["macro_f1"]:.6f} recall={row["macro_recall"]:.6f} TP={row["total_tp"]} FP={row["total_fp"]} constraints={bool(row["constraints_met"])}',flush=True)
 # Select one illustrative feasible Pareto candidate using ONLY exploration data.
 # This is a *development hypothesis*, not a frozen independently validated policy.
 candidates=[r for r in frontier if r['feasible']]
 candidate=max(candidates,key=lambda r:(r['macro_f1'],-r['total_fp'])) if candidates else None
 if candidate:
  cs=float(candidate['sparse_cutoff']);cd=float(candidate['dense_cutoff'])
  for split,subset in [('exploration',fit),('development_check',check)]:
   stats=aggregate(subset,cs,cd);g=aggregate(subset,frozen,frozen)
   print(f'{split} illustrative candidate: sparse={cs:.6f} dense={cd:.6f} F1={objective(stats):.6f} FP={sum(v["fp"] for v in stats.values())} constraints={feasible(stats,g,a.max_fp_ratio,a.reference_f1_tolerance)}',flush=True)
 audit={'experiment':'E5-C66','status':'EXPLORATORY DEVELOPMENT ONLY; NO INDEPENDENT GENERALIZATION',
        'exploration_seeds':list(FIT),'development_check_seeds':list(CHECK),
        'scenarios':SCENARIOS,'density_boundary':16,
        'max_fp_ratio_per_scenario':a.max_fp_ratio,'reference_f1_tolerance':a.reference_f1_tolerance,
        'grid_points':len(curve),'pareto_points':len(frontier),
        'feasible_pareto_points':len(candidates),
        'illustrative_candidate':{'sparse_cutoff':candidate['sparse_cutoff'],'dense_cutoff':candidate['dense_cutoff']} if candidate else None,
        'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),
                        'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},
        'limitations':['All data use the same synthetic generator family',
                       'E5-C63 and E5-C65 were already inspected and must not be reused for new threshold selection',
                       'Pareto points depend on the predefined grid and finite development seeds',
                       'Any selected E5-C66 candidate requires untouched independent validation']}
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'threshold_grid.csv',curve)
 write_csv(a.output/'pareto_frontier.csv',frontier)
 write_csv(a.output/'frozen_policy_comparison.csv',comparisons)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('\nE5-C66 | DEVELOPMENT PARETO FRONTIER')
 print(f'Grid={len(curve)} Pareto={len(frontier)} Feasible-Pareto={len(candidates)}')
 print('Saved:',a.output.resolve())
 print('Development only. Do not use E5-C63/E5-C65 to select a new policy.')
if __name__=='__main__':main()
