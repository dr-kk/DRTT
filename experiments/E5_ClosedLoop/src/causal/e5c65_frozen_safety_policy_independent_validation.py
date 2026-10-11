"""E5-C65: independent frozen safety-constrained gate evaluation.

Compares original baseline, global gate, E5-C62 density gate, and E5-C64 constrained gate.
No model fitting except reconstruction of the E5-C15-trained detector; no tuning.
Shift axes are isolated: background count, campaign count, and observation
window duration. Duration is NOT a direct manipulation of attack inter-stage gaps.
Pickle inputs must be trusted local artifacts.
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

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def scores_to_metrics(rows,probs,cutoff,truth):
 thresholds=np.full(len(rows),float(cutoff)) if np.isscalar(cutoff) else np.asarray(cutoff,dtype=float)
 if len(thresholds)!=len(rows):raise ValueError('Threshold count mismatch')
 edges={(r['parent'],r['child']) for r,p,t in zip(rows,probs,thresholds) if float(p)>=float(t)}
 tp=len(edges&truth);fp=len(edges-truth);fn=len(truth-edges)
 return {'tp':tp,'fp':fp,'fn':fn,'predicted':len(edges),
         'precision':tp/(tp+fp) if tp+fp else 0.,
         'recall':tp/(tp+fn) if tp+fn else 0.,
         'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def pooled(rows,prefix):
 tp=sum(int(r[prefix+'_tp']) for r in rows)
 fp=sum(int(r[prefix+'_fp']) for r in rows)
 fn=sum(int(r[prefix+'_fn']) for r in rows)
 return {'tp':tp,'fp':fp,'fn':fn,
         'precision':tp/(tp+fp) if tp+fp else 0.,
         'recall':tp/(tp+fn) if tp+fn else 0.,
         'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('ranker','retrained','gate','compatible-gate'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--policy',type=Path,required=True,help='E5-C64 constrained policy')
 p.add_argument('--density-policy',type=Path,required=True,help='E5-C62 density policy')
 p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 p.add_argument('--scripts',type=Path,default=Path('src/causal'))
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seed-start',type=int,default=6500)
 p.add_argument('--num-seeds',type=int,default=10)
 a=p.parse_args()
 if a.num_seeds<2:p.error('--num-seeds must be >=2 for seed-level bootstrap')
 if a.output.exists():p.error('Output path exists; refusing to overwrite')
 for path in (a.ranker,a.retrained,a.gate,a.compatible_gate,a.policy,a.density_policy,a.train/'telemetry.csv',a.train/'ground_truth.csv'):
  if not path.is_file():p.error(f'Missing input: {path}')
 sys.path.insert(0,str(a.scripts.resolve()))
 import e5c15_reranker as base
 import e5c38_end_to_end as c38
 import e5c5_overlapping_generator as gen
 import e5c58_compatible_gate_calibration as c58
 with a.ranker.open('rb') as f:original=pickle.load(f)
 with a.retrained.open('rb') as f:modified=pickle.load(f)
 with a.gate.open('rb') as f:original_gate=pickle.load(f)
 with a.compatible_gate.open('rb') as f:modified_gate=pickle.load(f)
 if original['features']!=base.FEATURES or modified['features']!=base.FEATURES[:-1]:raise ValueError('Ranker feature mismatch')
 if original_gate['features']!=c58.GATE_FEATURES or modified_gate['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature mismatch')
 if modified.get('baseline_sha256')!=sha(a.ranker):raise ValueError('Modified ranker lineage mismatch')
 if original_gate.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Original gate lineage mismatch')
 if modified_gate.get('e5c56_sha256')!=sha(a.retrained) or modified_gate.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('Modified gate lineage mismatch')
 policy=json.loads(a.policy.read_text(encoding='utf-8'))
 density_policy=json.loads(a.density_policy.read_text(encoding='utf-8'))
 if policy.get('experiment')!='E5-C64' or policy.get('status')!='DEVELOPMENT CALIBRATION ONLY; NOT INDEPENDENT TEST':raise ValueError('Expected frozen E5-C64 policy')
 if density_policy.get('experiment')!='E5-C62':raise ValueError('Expected frozen E5-C62 policy')
 if policy.get('selection_seeds')!=list(range(6400,6405)) or policy.get('development_check_seeds')!=list(range(6405,6410)):raise ValueError('Unexpected E5-C64 provenance')
 if density_policy.get('selection_seeds')!=list(range(6200,6205)) or density_policy.get('development_check_seeds')!=list(range(6205,6210)):raise ValueError('Unexpected E5-C62 provenance')
 if int(policy['density_boundary'])!=16 or int(density_policy['density_boundary'])!=16:raise ValueError('Unexpected density boundary')
 if abs(float(policy['sparse_cutoff'])-.725)>1e-9 or abs(float(policy['dense_cutoff'])-.75)>1e-9:raise ValueError('Unexpected E5-C64 cutoffs')
 if abs(float(density_policy['sparse_cutoff'])-.55)>1e-9 or abs(float(density_policy['dense_cutoff'])-.8)>1e-9:raise ValueError('Unexpected E5-C62 cutoffs')
 if abs(float(policy['max_fp_ratio_per_scenario'])-1.25)>1e-9 or abs(float(policy['reference_f1_tolerance'])-.005)>1e-9:raise ValueError('Unexpected constraint parameters')
 expected_hashes={'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)}
 if policy.get('model_sha256')!=expected_hashes or density_policy.get('model_sha256')!=expected_hashes:raise ValueError('Policy model SHA256 mismatch')
 if abs(float(policy['original_global_cutoff'])-float(modified_gate['cutoff']))>1e-10:raise ValueError('Global cutoff mismatch')
 if abs(float(density_policy['original_global_cutoff'])-float(modified_gate['cutoff']))>1e-10:raise ValueError('Density policy global cutoff mismatch')
 used=set(range(5900,5910))|set(range(6000,6005))|set(range(6200,6210))|set(range(6300,6310))|set(range(6400,6410))
 if set(range(a.seed_start,a.seed_start+a.num_seeds)) & used:raise ValueError('Evaluation seeds overlap prior studies')
 if float(original['threshold'])!=.65 or int(original['window'])!=90 or int(modified['window'])!=90 or int(modified_gate['window'])!=90:raise ValueError('Frozen settings mismatch')
 budget=float(modified_gate['budget'])
 if abs(budget-1.5)>1e-9:raise ValueError('Expected frozen budget 1.5x')
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train]
 x=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([x[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 per=[];summary=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  sr=[]
  for seed in range(a.seed_start,a.seed_start+a.num_seeds):
   events,truthrows=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
   event_ids=[r['event_id'] for r in events]
   feats=c38.extract(events,90)
   probs=detector.predict_proba(np.asarray([feats[i] for i in event_ids]))[:,1]
   base_n=sum(float(r['signal'])>=.65 for r in events)
   n=min(len(events),int(round(base_n*budget)))
   chosen={i for i,_ in sorted(zip(event_ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
   old_rows=c58.top_rows(events,chosen,original,90)
   new_rows=c58.top_rows(events,chosen,modified,90)
   if [r['child'] for r in old_rows]!=[r['child'] for r in new_rows]:raise RuntimeError(f'Candidate-child mismatch: {scenario} seed={seed}')
   old_probs=original_gate['model'].predict_proba(np.asarray([r['features'] for r in old_rows]))[:,1] if old_rows else []
   new_probs=modified_gate['model'].predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1] if new_rows else []
   cnt=np.asarray([int(round(np.expm1(r['features'][4]))) for r in new_rows],dtype=int)
   if np.any(cnt<1):raise RuntimeError('Invalid candidate count')
   thresholds=np.where(cnt>=16,float(policy['dense_cutoff']),float(policy['sparse_cutoff']))
   density_thresholds=np.where(cnt>=16,float(density_policy['dense_cutoff']),float(density_policy['sparse_cutoff']))
   b=scores_to_metrics(old_rows,old_probs,float(original_gate['cutoff']),truth)
   q=scores_to_metrics(new_rows,new_probs,float(modified_gate['cutoff']),truth)
   d=scores_to_metrics(new_rows,new_probs,thresholds,truth)
   x62=scores_to_metrics(new_rows,new_probs,density_thresholds,truth)
   row={'scenario':scenario,'seed':seed,'campaigns':campaigns,'duration':duration,'background':background,'events':len(events),'alerts':n,'truth_edges':len(truth),'candidate_children':len(old_rows)}
   row.update({'baseline_'+k:v for k,v in b.items()})
   row.update({'proposed_'+k:v for k,v in q.items()})
   row.update({'density_'+k:v for k,v in d.items()})
   row.update({'e5c62_'+k:v for k,v in x62.items()})
   row['delta_f1']=d['f1']-q['f1'];row['delta_fp']=d['fp']-q['fp'];row['fp_ratio']=d['fp']/q['fp'] if q['fp'] else float('inf')
   per.append(row);sr.append(row)
   print(f'{scenario} seed={seed}: truth={len(truth)} alerts={n} baseline TP/FP={b["tp"]}/{b["fp"]} F1={b["f1"]:.4f} global TP/FP={q["tp"]}/{q["fp"]} F1={q["f1"]:.4f} E5C64 TP/FP={d["tp"]}/{d["fp"]} F1={d["f1"]:.4f} delta={row["delta_f1"]:+.4f}',flush=True)
  b=pooled(sr,'baseline');q=pooled(sr,'proposed');d=pooled(sr,'density');x62=pooled(sr,'e5c62')
  rng=np.random.default_rng(600060+list(SCENARIOS).index(scenario))
  ds=[]
  for _ in range(5000):
   sampled=[sr[int(i)] for i in rng.integers(0,len(sr),len(sr))]
   ds.append(pooled(sampled,'density')['f1']-pooled(sampled,'proposed')['f1'])
  lo,hi=np.percentile(ds,[2.5,97.5])
  r={'scenario':scenario,'campaigns':campaigns,'duration':duration,'background':background,'seeds':len(sr),'truth_edges':sum(z['truth_edges'] for z in sr),
     'baseline_tp':b['tp'],'baseline_fp':b['fp'],'baseline_precision':b['precision'],'baseline_recall':b['recall'],'baseline_f1':b['f1'],
     'proposed_tp':q['tp'],'proposed_fp':q['fp'],'proposed_precision':q['precision'],'proposed_recall':q['recall'],'proposed_f1':q['f1'],
     'e5c62_f1':x62['f1'],'e5c62_fp':x62['fp'],'e5c62_tp':x62['tp'],'density_tp':d['tp'],'density_fp':d['fp'],'density_precision':d['precision'],'density_recall':d['recall'],'density_f1':d['f1'],
     'delta_f1':d['f1']-q['f1'],'delta_fp':d['fp']-q['fp'],'fp_ratio':d['fp']/q['fp'] if q['fp'] else float('inf'),'fp_constraint_met':int(d['fp']<=float(policy['max_fp_ratio_per_scenario'])*q['fp']+1e-9),'f1_delta_ci_low':float(lo),'f1_delta_ci_high':float(hi),
     'seed_positive':sum(z['delta_f1']>0 for z in sr),'seed_zero':sum(z['delta_f1']==0 for z in sr),'seed_negative':sum(z['delta_f1']<0 for z in sr)}
  summary.append(r)
  print(f'SUMMARY {scenario}: global={q["f1"]:.4f} E5C62={x62["f1"]:.4f} E5C64={d["f1"]:.4f} delta={r["delta_f1"]:+.4f} CI=[{lo:+.4f},{hi:+.4f}] FP ratio={r["fp_ratio"]:.3f} constraint={bool(r["fp_constraint_met"])}',flush=True)
 audit={'experiment':'E5-C65','status':'INDEPENDENT FROZEN SAFETY-CONSTRAINED POLICY EVALUATION','seed_range':[a.seed_start,a.seed_start+a.num_seeds-1],
        'alert_budget':budget,'policy_sha256':sha(a.policy),'e5c62_policy_sha256':sha(a.density_policy),'policy_thresholds':{'sparse':policy['sparse_cutoff'],'dense':policy['dense_cutoff'],'boundary':policy['density_boundary']},'scenarios':{k:{'campaigns':v[0],'duration':v[1],'background_events':v[2]} for k,v in SCENARIOS.items()},
        'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},
        'train_sha256':{'telemetry':sha(a.train/'telemetry.csv'),'ground_truth':sha(a.train/'ground_truth.csv')},
        'limitations':['All conditions use same synthetic generator family; not real-world domain shift','The same seed is reused across scenarios for paired comparisons, but event streams are regenerated','Duration changes observation-window density, not the 8-25 unit attack-stage gap distribution','No model fitting beyond rebuilding the frozen-training detector; no threshold selection on test seeds','Seed-level bootstrap CIs are approximate with few seeds; seven scenario comparisons are exploratory']}
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'per_seed.csv',per)
 write_csv(a.output/'scenario_summary.csv',summary)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('\nE5-C65 | FROZEN FOUR-PIPELINE INDEPENDENT EVALUATION')
 for r in summary:print(f'{r["scenario"]}: F1 baseline={r["baseline_f1"]:.4f} global={r["proposed_f1"]:.4f} density={r["density_f1"]:.4f} constrained-global={r["delta_f1"]:+.4f} FP global/E5C62/E5C64={r["proposed_fp"]}/{r["e5c62_fp"]}/{r["density_fp"]} constraint={bool(r["fp_constraint_met"])}')
 # Macro-average scenario-level F1, and pooled totals across all scenarios.
 macro={k:float(np.mean([r[k+'_f1'] for r in summary])) for k in ('baseline','proposed','e5c62','density')}
 total={k:pooled(per,k) for k in ('baseline','proposed','e5c62','density')}
 (a.output/'overall_summary.json').write_text(json.dumps({'macro_scenario_f1':macro,'pooled_all_scenarios':total,'constrained_vs_global_macro_delta':macro['density']-macro['proposed'],'all_fp_constraints_met':all(r['fp_constraint_met'] for r in summary),'reference_f1_floor_met':next(r for r in summary if r['scenario']=='reference')['density_f1']>=next(r for r in summary if r['scenario']=='reference')['proposed_f1']-float(policy['reference_f1_tolerance'])},indent=2),encoding='utf-8')
 print('MACRO F1:',macro)
 print('ALL FP CONSTRAINTS MET:',all(r['fp_constraint_met'] for r in summary))
 print('ALL-SCENARIO POOLED:',total)
 print('Saved:',a.output.resolve())
if __name__=='__main__':main()
