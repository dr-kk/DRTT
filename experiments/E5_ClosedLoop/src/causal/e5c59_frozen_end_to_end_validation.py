"""E5-C59: frozen independent end-to-end synthetic validation (seeds 5900-5909).
No model fitting except the original E5-C15-trained detector; no threshold tuning.
All model pickles must be locally trusted artifacts.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(rows,probs,cutoff,truth):
    keep=np.asarray(probs)>=cutoff
    edges={(r['parent'],r['child']) for r,k in zip(rows,keep) if k}
    tp=len(edges & truth);fp=len(edges-truth);fn=len(truth-edges)
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'predicted':len(edges),'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--retrained',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--compatible-gate',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error('Output directory nonempty; refusing overwrite')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    import e5c38_end_to_end as c38
    import e5c5_overlapping_generator as gen
    import e5c58_compatible_gate_calibration as c58
    with a.ranker.open('rb') as f:baseline=pickle.load(f)
    with a.retrained.open('rb') as f:proposed=pickle.load(f)
    with a.gate.open('rb') as f:old_gate=pickle.load(f)
    with a.compatible_gate.open('rb') as f:new_gate=pickle.load(f)
    if baseline['features']!=base.FEATURES or proposed['features']!=base.FEATURES[:-1]:raise ValueError('Reranker feature mismatch')
    if old_gate['features']!=c58.GATE_FEATURES or new_gate['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature mismatch')
    if proposed.get('baseline_sha256')!=sha(a.ranker):raise ValueError('E5-C56 model lineage mismatch')
    if old_gate.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Original gate lineage mismatch')
    if new_gate.get('e5c56_sha256')!=sha(a.retrained) or new_gate.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('New gate lineage mismatch')
    if float(baseline['threshold'])!=.65 or int(baseline['window'])!=90 or int(proposed['window'])!=90 or int(new_gate['window'])!=90:raise ValueError('Frozen configuration mismatch')
    budget=float(new_gate['budget'])
    if abs(budget-1.5)>1e-9:raise ValueError('Expected preselected budget x1.5')
    for name in ('telemetry.csv','ground_truth.csv'):
        if not (a.train/name).is_file():ap.error(f'Missing training file: {a.train/name}')
    train_events=c58.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
    ids=[r['event_id'] for r in train_events]
    feats=c38.extract(train_events,90)
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    per=[]
    for seed in range(5900,5910):
        events,truthrows=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        event_ids=[r['event_id'] for r in events]
        x=c38.extract(events,90)
        scores=detector.predict_proba(np.asarray([x[i] for i in event_ids]))[:,1]
        base_n=sum(float(r['signal'])>=.65 for r in events)
        alert_n=min(len(event_ids),int(round(base_n*budget)))
        chosen={i for i,_ in sorted(zip(event_ids,scores),key=lambda z:(-float(z[1]),z[0]))[:alert_n]}
        old_rows=c58.top_rows(events,chosen,baseline,90)
        new_rows=c58.top_rows(events,chosen,proposed,90)
        if [r['child'] for r in old_rows]!=[r['child'] for r in new_rows]:raise RuntimeError(f'Seed {seed}: candidate child mismatch')
        old_probs=old_gate['model'].predict_proba(np.asarray([r['features'] for r in old_rows]))[:,1] if old_rows else np.array([])
        new_probs=new_gate['model'].predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1] if new_rows else np.array([])
        bm=metrics(old_rows,old_probs,float(old_gate['cutoff']),truth)
        pm=metrics(new_rows,new_probs,float(new_gate['cutoff']),truth)
        row={'seed':seed,'alerts':alert_n,'true_edges':len(truth),'candidate_children':len(old_rows)}
        row.update({f'baseline_{k}':v for k,v in bm.items()})
        row.update({f'proposed_{k}':v for k,v in pm.items()})
        row['delta_f1']=pm['f1']-bm['f1']
        row['delta_fp']=pm['fp']-bm['fp']
        per.append(row)
        print(f'Seed {seed}: alerts={alert_n} baseline TP/FP={bm["tp"]}/{bm["fp"]} F1={bm["f1"]:.4f} proposed TP/FP={pm["tp"]}/{pm["fp"]} F1={pm["f1"]:.4f} delta={row["delta_f1"]:+.4f}',flush=True)
    def pooled(prefix,rows):
        tp=sum(int(r[prefix+'_tp']) for r in rows);fp=sum(int(r[prefix+'_fp']) for r in rows);fn=sum(int(r[prefix+'_fn']) for r in rows)
        return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    b=pooled('baseline',per);p=pooled('proposed',per)
    rng=np.random.default_rng(590059)
    deltas=[]
    for _ in range(10000):
        sample=[per[int(i)] for i in rng.integers(0,len(per),len(per))]
        deltas.append(pooled('proposed',sample)['f1']-pooled('baseline',sample)['f1'])
    ci=np.percentile(deltas,[2.5,97.5]).tolist()
    audit={'experiment':'E5-C59','status':'INDEPENDENT FROZEN SYNTHETIC EVALUATION','seeds':list(range(5900,5910)),'alert_budget':budget,'baseline':b,'proposed':p,'pooled_f1_delta':p['f1']-b['f1'],'paired_seed_bootstrap_95ci_f1_delta':ci,'seeds_positive_zero_negative':[sum(r['delta_f1']>0 for r in per),sum(r['delta_f1']==0 for r in per),sum(r['delta_f1']<0 for r in per)],'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},'train_sha256':{'telemetry':sha(a.train/'telemetry.csv'),'ground_truth':sha(a.train/'ground_truth.csv')},'limitations':['Synthetic data from the same generator family','Ten seeds only; seed-level bootstrap CI is approximate','No real operational causal ground truth','Detector trained on E5-C15 training split; both pipelines use identical alerts','No hyperparameter changes or threshold selection on evaluation seeds']}
    a.output.mkdir(parents=True,exist_ok=False)
    write_csv(a.output/'per_seed.csv',per)
    (a.output/'summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('\nE5-C59 | FROZEN END-TO-END VALIDATION')
    print('Baseline:',json.dumps(b))
    print('Proposed:',json.dumps(p))
    print(f'Pooled F1 delta: {audit["pooled_f1_delta"]:+.6f}; 95% paired seed-bootstrap CI [{ci[0]:+.6f}, {ci[1]:+.6f}]')
    print('Saved:',a.output.resolve())
if __name__=='__main__':main()
