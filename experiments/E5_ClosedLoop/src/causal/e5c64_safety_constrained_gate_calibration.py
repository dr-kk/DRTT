"""E5-C64: safety-constrained density gate calibration, development only.

Select on synthetic seeds 6400-6404; check on 6405-6409. No reuse of E5-C63
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
FIT=range(6400,6405)
CHECK=range(6405,6410)

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
 a=p.parse_args()
 if a.output.exists():p.error('Output exists: refusing overwrite')
 if a.max_fp_ratio<1:p.error('max-fp-ratio must be >=1; global fallback must remain feasible')
 if a.reference_f1_tolerance<0:p.error('reference-f1-tolerance must be nonnegative')
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
 # Grid predeclared. Include frozen global policy to guarantee a feasible fallback.
 grid=sorted(set([round(float(x),3) for x in np.arange(.30,.951,.025)]+[frozen]))
 curve=[];best=None;best_key=None
 for cs in grid:
  for cd in grid:
   st=aggregate(fit,cs,cd)
   ok=feasible(st,global_fit,a.max_fp_ratio,a.reference_f1_tolerance)
   score=objective(st)
   row={'sparse_cutoff':cs,'dense_cutoff':cd,'macro_f1':score,'feasible':int(ok),
        'reference_f1':st['reference']['f1'],
        'background_half_fp_ratio':st['background_half']['fp']/global_fit['background_half']['fp'] if global_fit['background_half']['fp'] else 0.,
        'duration_double_fp_ratio':st['duration_double']['fp']/global_fit['duration_double']['fp'] if global_fit['duration_double']['fp'] else 0.}
   curve.append(row)
   key=(score,-abs(cs-frozen)-abs(cd-frozen))
   if ok and (best is None or key>best_key):best=row;best_key=key
 if best is None:raise RuntimeError('No feasible policy; global fallback should be feasible')
 cs=float(best['sparse_cutoff']);cd=float(best['dense_cutoff'])
 summary=[]
 for split,subset in [('selection',fit),('development_check',check)]:
  global_st=aggregate(subset,frozen,frozen)
  chosen_st=aggregate(subset,cs,cd)
  for scenario in SCENARIOS:
   g=global_st[scenario];d=chosen_st[scenario]
   summary.append({'split':split,'scenario':scenario,'global_f1':g['f1'],'constrained_f1':d['f1'],
    'delta_f1':d['f1']-g['f1'],'global_tp':g['tp'],'constrained_tp':d['tp'],
    'global_fp':g['fp'],'constrained_fp':d['fp'],
    'fp_ratio':d['fp']/g['fp'] if g['fp'] else 0.,'fp_constraint_met':int(d['fp']<=a.max_fp_ratio*g['fp']+1e-9)})
   print(f'{split} {scenario}: F1 {g["f1"]:.4f}->{d["f1"]:.4f} delta={d["f1"]-g["f1"]:+.4f} TP {g["tp"]}->{d["tp"]} FP {g["fp"]}->{d["fp"]}',flush=True)
  print(f'{split} macro F1: {objective(global_st):.6f}->{objective(chosen_st):.6f}; '
        f'constraints_met={feasible(chosen_st,global_st,a.max_fp_ratio,a.reference_f1_tolerance)}',flush=True)
 policy={'experiment':'E5-C64','status':'DEVELOPMENT CALIBRATION ONLY; NOT INDEPENDENT TEST',
  'selection_seeds':list(FIT),'development_check_seeds':list(CHECK),
  'scenarios':SCENARIOS,'density_boundary':16,'sparse_cutoff':cs,'dense_cutoff':cd,
  'original_global_cutoff':frozen,'max_fp_ratio_per_scenario':a.max_fp_ratio,
  'reference_f1_tolerance':a.reference_f1_tolerance,
  'selection_objective':'maximize macro scenario pooled F1 subject to per-scenario FP <= ratio*global FP and reference F1 floor',
  'selection_feasible_grid_points':sum(r['feasible'] for r in curve),
  'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),
   'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},
  'limitations':['Safety means relative synthetic false-attribution count, not verified operational safety',
   'Development-check constraints may fail even if selection constraints hold',
   'E5-C63 findings motivated this hypothesis; E5-C63 is not a valid independent test of E5-C64',
   'A new untouched independent evaluation is required after policy freeze',
   'All scenarios use same synthetic generator family; no real-world evidence']}
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'threshold_grid.csv',curve)
 write_csv(a.output/'development_summary.csv',summary)
 (a.output/'constrained_gate_policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
 print('\nE5-C64 | SAFETY-CONSTRAINED DEVELOPMENT CALIBRATION')
 print(f'Frozen global cutoff: {frozen:.8f}')
 print(f'Selected sparse cutoff (<16): {cs:.8f}')
 print(f'Selected dense cutoff (>=16): {cd:.8f}')
 print(f'Feasible threshold pairs: {policy["selection_feasible_grid_points"]}/{len(curve)}')
 print('Saved:',a.output.resolve())
 print('No independent generalization claim. Do not reuse E5-C63 for validation.')
if __name__=='__main__':main()
