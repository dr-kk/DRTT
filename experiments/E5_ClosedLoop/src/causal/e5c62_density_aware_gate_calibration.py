"""E5-C62: density-aware gate cutoff calibration on NEW development seeds.

Threshold fit: 6200-6204; development check: 6205-6209.
No reuse of E5-C59/60 test seeds; no model retraining. E5-C63 remains untouched.
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
 'duration_half':(100,1200,10000),
 'duration_double':(100,4800,10000),
}
DENSITY_BOUNDARY=16  # >=16 parents: dense; <=15: sparse. Predeclared.
SELECTION_SEEDS=range(6200,6205)
CHECK_SEEDS=range(6205,6210)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def counts(probs,truth_flags,dense,cut_sparse,cut_dense,truth_n):
 keep=probs>=np.where(dense,cut_dense,cut_sparse)
 tp=int(np.count_nonzero(keep & truth_flags));fp=int(np.count_nonzero(keep & ~truth_flags))
 fn=truth_n-tp
 return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def aggregate(records,cs,cd):
 stats={}
 for scenario in SCENARIOS:
  subset=[r for r in records if r['scenario']==scenario]
  tp=fp=fn=0
  for r in subset:
   m=counts(r['probs'],r['correct'],r['dense'],cs,cd,r['truth_n'])
   tp+=m['tp'];fp+=m['fp'];fn+=m['fn']
  stats[scenario]={'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
 return stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('ranker','retrained','gate','compatible-gate'):
  p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 p.add_argument('--scripts',type=Path,default=Path('src/causal'))
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Output already exists: refusing overwrite')
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
 if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Expected frozen alert budget 1.5x')
 if abs(float(ng['cutoff'])-0.74376355)>1e-6:raise ValueError('Unexpected frozen gate cutoff')
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train];features=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([features[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 records=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  for seed in list(SELECTION_SEEDS)+list(CHECK_SEEDS):
   events,ground=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
   event_ids=[r['event_id'] for r in events]
   f=c38.extract(events,90)
   probs=detector.predict_proba(np.asarray([f[i] for i in event_ids]))[:,1]
   alert_n=min(len(events),int(round(sum(float(r['signal'])>=.65 for r in events)*1.5)))
   chosen={i for i,_ in sorted(zip(event_ids,probs),key=lambda z:(-float(z[1]),z[0]))[:alert_n]}
   old_rows=c58.top_rows(events,chosen,old,90)
   new_rows=c58.top_rows(events,chosen,new,90)
   if [r['child'] for r in old_rows]!=[r['child'] for r in new_rows]:raise RuntimeError('Candidate children mismatch')
   if not new_rows:raise RuntimeError('No candidate children')
   gate_probs=np.asarray(ng['model'].predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1],dtype=float)
   # log_candidate_count is the fifth gate feature; recover exact integer via round.
   cnt=np.asarray([int(round(np.expm1(r['features'][4]))) for r in new_rows],dtype=int)
   if np.any(cnt<1):raise RuntimeError('Invalid candidate count')
   correct=np.asarray([(r['parent'],r['child']) in truth for r in new_rows],dtype=bool)
   records.append({'scenario':scenario,'seed':seed,'truth_n':len(truth),'probs':gate_probs,'correct':correct,'dense':cnt>=DENSITY_BOUNDARY,'alerts':alert_n})
   print(f'{scenario} seed={seed} alerts={alert_n} candidates={len(new_rows)} dense={int(np.sum(cnt>=DENSITY_BOUNDARY))}',flush=True)
 fit=[r for r in records if r['seed'] in SELECTION_SEEDS]
 check=[r for r in records if r['seed'] in CHECK_SEEDS]
 frozen=float(ng['cutoff'])
 # Finite predeclared threshold grid, with existing cutoff included. No test-set access.
 grid=sorted(set([round(float(x),3) for x in np.arange(.30,.951,.025)]+[frozen]))
 # Select on macro-average of scenario pooled F1, to prevent dense scenarios dominating.
 curve=[]
 for cs in grid:
  for cd in grid:
   st=aggregate(fit,cs,cd)
   score=float(np.mean([v['f1'] for v in st.values()]))
   curve.append({'sparse_cutoff':cs,'dense_cutoff':cd,'macro_scenario_f1':score,
                 'reference_f1':st['reference']['f1'],
                 'background_half_f1':st['background_half']['f1'],
                 'duration_double_f1':st['duration_double']['f1']})
 # Tie-break toward smaller deviation from frozen global cutoff.
 best=max(curve,key=lambda r:(r['macro_scenario_f1'],-abs(r['sparse_cutoff']-frozen)-abs(r['dense_cutoff']-frozen)))
 chosen_cs=float(best['sparse_cutoff']);chosen_cd=float(best['dense_cutoff'])
 summary=[]
 for split,subset in [('selection',fit),('development_check',check)]:
  frozen_stats=aggregate(subset,frozen,frozen)
  chosen_stats=aggregate(subset,chosen_cs,chosen_cd)
  for scenario in SCENARIOS:
   b=frozen_stats[scenario];q=chosen_stats[scenario]
   summary.append({'split':split,'scenario':scenario,'frozen_f1':b['f1'],'density_aware_f1':q['f1'],
    'delta_f1':q['f1']-b['f1'],'frozen_tp':b['tp'],'density_aware_tp':q['tp'],
    'frozen_fp':b['fp'],'density_aware_fp':q['fp']})
   print(f'{split} {scenario}: F1 {b["f1"]:.4f}->{q["f1"]:.4f} delta={q["f1"]-b["f1"]:+.4f} TP {b["tp"]}->{q["tp"]} FP {b["fp"]}->{q["fp"]}')
 check_delta=float(np.mean([r['delta_f1'] for r in summary if r['split']=='development_check']))
 obj={'experiment':'E5-C62','status':'DEVELOPMENT CALIBRATION ONLY; NOT INDEPENDENT TEST',
      'selection_seeds':list(SELECTION_SEEDS),'development_check_seeds':list(CHECK_SEEDS),
      'scenarios':SCENARIOS,'density_boundary':DENSITY_BOUNDARY,
      'sparse_definition':'candidate_count <16','dense_definition':'candidate_count >=16',
      'sparse_cutoff':chosen_cs,'dense_cutoff':chosen_cd,'original_global_cutoff':frozen,
      'selection_objective':'macro-average scenario-level pooled edge F1',
      'development_check_macro_f1_delta':check_delta,
      'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),
       'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},
      'limitations':['No model fitting beyond reconstruction of the frozen detector on E5-C15 training data',
       'Threshold selection uses ground truth on seeds 6200-6204; optimistic selection performance',
       'Seeds 6205-6209 are development check only, not final independent test',
       'Five conditions and 2D threshold search can overfit to synthetic generator family',
       'Previously observed E5-C60 sparse failures motivated this hypothesis; E5-C63 must use untouched seeds',
       'Candidate count is a proxy for ambiguity, not proof of causal density effects']}
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'threshold_grid.csv',curve)
 write_csv(a.output/'development_summary.csv',summary)
 (a.output/'density_gate_policy.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
 print('\nE5-C62 | DEVELOPMENT DENSITY-AWARE CALIBRATION')
 print(f'Frozen global cutoff: {frozen:.8f}')
 print(f'Selected sparse cutoff (<16): {chosen_cs:.8f}')
 print(f'Selected dense cutoff (>=16): {chosen_cd:.8f}')
 print(f'Development-check macro F1 delta: {check_delta:+.6f}')
 print('Saved:',a.output.resolve())
 print('No independent generalization claim. Next: E5-C63 on untouched seeds.')
if __name__=='__main__':main()
