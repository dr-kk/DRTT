"""E5-C69: final frozen independent test on untouched seeds 6900-6909.
No model training changes, threshold selection, or further iteration.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCENARIOS={'reference':(100,2400,10000),'background_half':(100,2400,5000),
'background_double':(100,2400,20000),'campaigns_half':(50,2400,10000),
'campaigns_onehalf':(150,2400,10000),'duration_half':(100,1200,10000),
'duration_double':(100,4800,10000)}
SEEDS=range(6900,6910)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def metrics(tp,fp,n):
 return {'tp':tp,'fp':fp,'fn':n-tp,'recall':tp/n if n else 0.,
         'precision':tp/(tp+fp) if tp+fp else 0.,
         'f1':2*tp/(n+tp+fp) if n+tp+fp else 0.}

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('ranker','retrained','gate','compatible-gate','safety-policy','candidate-summary'):
  p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 p.add_argument('--scripts',type=Path,default=Path('src/causal'))
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-fp-ratio',type=float,default=1.25)
 p.add_argument('--reference-f1-tolerance',type=float,default=.005)
 p.add_argument('--cost-ratio',type=float,default=0.1,help='Predeclared C_FP/C_FN; >0.0516383 selects conservative')
 a=p.parse_args()
 if a.output.exists():p.error('Output exists; refusing overwrite')
 if a.cost_ratio < 0: p.error('Cost ratio must be nonnegative')
 for path in (a.ranker,a.retrained,a.gate,a.compatible_gate,a.safety_policy,a.candidate_summary,a.train/'telemetry.csv',a.train/'ground_truth.csv'):
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
 if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1]:raise ValueError('Ranker feature mismatch')
 if og['features']!=c58.GATE_FEATURES or ng['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature mismatch')
 if new.get('baseline_sha256')!=sha(a.ranker) or og.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Ranker lineage mismatch')
 if ng.get('e5c56_sha256')!=sha(a.retrained) or ng.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('Gate lineage mismatch')
 if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Unexpected alert budget')
 global_cutoff=float(ng['cutoff'])
 safety=json.loads(a.safety_policy.read_text(encoding='utf-8'))
 if safety.get('experiment')!='E5-C64' or safety.get('model_sha256',{}).get('proposed_ranker')!=sha(a.retrained) or safety.get('model_sha256',{}).get('compatible_gate')!=sha(a.compatible_gate):
  raise ValueError('E5-C64 policy provenance mismatch')
 if int(safety['density_boundary'])!=16 or abs(float(safety['original_global_cutoff'])-global_cutoff)>1e-9:raise ValueError('Safety policy config mismatch')
 summary=json.loads(a.candidate_summary.read_text(encoding='utf-8'))
 if summary.get('experiment')!='E5-C66A' or summary.get('total_grid_points')!=784:raise ValueError('Unexpected candidate summary')
 best=summary['best_feasible_by_macro_f1']
 if not best or int(best['feasible'])!=1:raise ValueError('No feasible candidate')
 cs,cd=float(best['sparse_cutoff']),float(best['dense_cutoff'])
 if abs(cs-.725)>1e-9 or abs(cd-.8)>1e-9:raise ValueError('Candidate differs from frozen E5-C66A output')
 policies={'Global':(global_cutoff,global_cutoff),
           'E5-C64':(float(safety['sparse_cutoff']),float(safety['dense_cutoff'])),
           'E5-C66A':(cs,cd)}
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train]
 feats=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 totals={name:{scenario:[0,0,0] for scenario in SCENARIOS} for name in policies}
 seed_rows=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  for seed in SEEDS:
   events,ground=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
   event_ids=[r['event_id'] for r in events]
   ef=c38.extract(events,90)
   probabilities=detector.predict_proba(np.asarray([ef[i] for i in event_ids]))[:,1]
   n=min(len(events),int(round(sum(float(r['signal'])>=.65 for r in events)*1.5)))
   chosen={i for i,_ in sorted(zip(event_ids,probabilities),key=lambda z:(-float(z[1]),z[0]))[:n]}
   rows=c58.top_rows(events,chosen,new,90)
   if not rows:raise ValueError('No candidate children')
   gp=np.asarray(ng['model'].predict_proba(np.asarray([r['features'] for r in rows]))[:,1],dtype=float)
   cnt=np.asarray([int(round(np.expm1(r['features'][4]))) for r in rows],dtype=int)
   if np.any(cnt<1):raise ValueError('Invalid candidate count')
   dense=cnt>=16
   correct=np.asarray([(r['parent'],r['child']) in truth for r in rows],dtype=bool)
   for name,(s,d) in policies.items():
    keep=gp>=np.where(dense,d,s)
    tp=int(np.count_nonzero(keep&correct));fp=int(np.count_nonzero(keep&~correct))
    t=totals[name][scenario];t[0]+=tp;t[1]+=fp;t[2]+=len(truth)
    seed_rows.append({'scenario':scenario,'seed':seed,'policy':name,**metrics(tp,fp,len(truth))})
   print(f'{scenario} seed={seed}: truth={len(truth)} alerts={n} candidates={len(rows)}',flush=True)
 scenario_rows=[]
 for scenario in SCENARIOS:
  g=metrics(*totals['Global'][scenario]);ref=None
  for name in policies:
   m=metrics(*totals[name][scenario])
   ratio=m['fp']/g['fp'] if g['fp'] else (1. if m['fp']==0 else float('inf'))
   row={'scenario':scenario,'policy':name,**m,'delta_f1_vs_global':m['f1']-g['f1'],
        'fp_ratio_vs_global':ratio,'fp_constraint_met':ratio<=a.max_fp_ratio+1e-12}
   scenario_rows.append(row)
   print(f'{scenario} {name}: F1={m["f1"]:.6f} TP={m["tp"]} FP={m["fp"]} FP_ratio={ratio:.3f}')
 global_reference=next(r for r in scenario_rows if r['scenario']=='reference' and r['policy']=='Global')['f1']
 summary_rows=[]
 for name in policies:
  subset=[r for r in scenario_rows if r['policy']==name]
  reference=next(r for r in subset if r['scenario']=='reference')
  all_fp=all(r['fp_constraint_met'] for r in subset)
  ref_ok=reference['f1']>=global_reference-a.reference_f1_tolerance-1e-12
  tp=sum(r['tp'] for r in subset);fp=sum(r['fp'] for r in subset);n=sum(r['tp']+r['fn'] for r in subset)
  row={'policy':name,'macro_f1':float(np.mean([r['f1'] for r in subset])),
       'macro_recall':float(np.mean([r['recall'] for r in subset])),
       'total_tp':tp,'total_fp':fp,'pooled_f1':metrics(tp,fp,n)['f1'],
       'all_fp_constraints_met':all_fp,'reference_constraint_met':ref_ok,
       'all_constraints_met':all_fp and ref_ok}
  row['weighted_loss']=row['total_tp']+row['total_fp']*0 if False else (n-tp)+a.cost_ratio*fp
  summary_rows.append(row)
  print(f'CHECK {name}: macro_F1={row["macro_f1"]:.6f} macro_recall={row["macro_recall"]:.6f} TP={tp} FP={fp} constraints={row["all_constraints_met"]}')
 selected='E5-C64' if a.cost_ratio<=249/4822 else 'E5-C66A'
 print(f'FROZEN COST RULE: ratio={a.cost_ratio:g} crossover={249/4822:.8f} selected={selected}')
 print('INDEPENDENT weighted losses:',{r['policy']:round(r['weighted_loss'],3) for r in summary_rows})
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'seed_metrics.csv',seed_rows)
 write_csv(a.output/'scenario_metrics.csv',scenario_rows)
 write_csv(a.output/'independent_summary.csv',summary_rows)
 audit={'experiment':'E5-C69','status':'FROZEN INDEPENDENT SYNTHETIC-GENERATOR VALIDATION; NO RETUNING',
        'seeds':list(SEEDS),'scenarios':SCENARIOS,'policies':policies,
        'candidate_source':str(a.candidate_summary),'candidate_selection_seeds':'6600-6604',
        'candidate_development_check_seeds':'6605-6609',
        'independent_test_seeds':'6900-6909','predeclared_cost_ratio':a.cost_ratio,
        'predeclared_crossover':249/4822,'predeclared_selected_mode':selected,
        'model_sha256':{'ranker':sha(a.retrained),'gate':sha(a.compatible_gate)},
        'caveats':['E5-C66A candidate was selected on 6600-6604',
                   '6900-6909 were reserved for independent evaluation',
                   'Do not tune after this test',
                   'All shifts use the same synthetic generator family; no real-world claim']}
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('\nE5-C69 | FINAL FROZEN INDEPENDENT VALIDATION')
 print('Saved:',a.output.resolve())
 print('Final independent synthetic test. Freeze results and proceed to manuscript.')
if __name__=='__main__':main()
