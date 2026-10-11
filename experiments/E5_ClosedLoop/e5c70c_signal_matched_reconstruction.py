"""E5-C70C: paired signal-matched robustness audit of frozen E5-C69 pipeline.
Run from D:\\DRTT\\E5_ClosedLoop. No tuning or modifications to E5-C69.
"""
import argparse, csv, hashlib, json, pickle, random, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()

def write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--retrained',type=Path,default=Path('results/causal/e5c56/reranker_no_gap_rank.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--compatible-gate',type=Path,default=Path('results/causal/e5c58/compatible_gate.pkl'))
    p.add_argument('--safety-policy',type=Path,default=Path('results/causal/e5c64/constrained_gate_policy.json'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c70c_signal_matched'))
    p.add_argument('--seeds',type=int,nargs='+',default=[7001,7002,7003,7004,7005])
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; refusing overwrite')
    required=[a.ranker,a.retrained,a.gate,a.compatible_gate,a.safety_policy,a.train/'telemetry.csv',a.train/'ground_truth.csv']
    for path in required:
        if not path.is_file():p.error('Missing input: '+str(path))
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
    if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Alert budget mismatch')
    safety=json.loads(a.safety_policy.read_text(encoding='utf-8'))
    if safety.get('experiment')!='E5-C64' or safety.get('model_sha256',{}).get('proposed_ranker')!=sha(a.retrained) or safety.get('model_sha256',{}).get('compatible_gate')!=sha(a.compatible_gate):raise ValueError('Policy lineage mismatch')
    if int(safety['density_boundary'])!=16 or abs(float(safety['original_global_cutoff'])-float(ng['cutoff']))>1e-9:raise ValueError('Policy configuration mismatch')
    policies={'Global':(float(ng['cutoff']),float(ng['cutoff'])), 'E5-C64':(float(safety['sparse_cutoff']),float(safety['dense_cutoff'])), 'E5-C66A':(.725,.800)}
    train=c58.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
    ids=[r['event_id'] for r in train]
    feats=c38.extract(train,90)
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    output=[]
    for seed in a.seeds:
        events,ground=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
        byid={r['event_id']:int(r['malicious']) for r in ground}
        benign_signals=[r['signal'] for r in events if not byid[r['event_id']]]
        rng=random.Random(170000+seed)
        matched=[dict(r) for r in events]
        # Replace EVERY event signal with independent draws from the original benign marginal.
        # This equalizes signal distributions without changing topology, timestamps, or truth.
        for r in matched:r['signal']=rng.choice(benign_signals)
        original_budget=min(len(events),int(round(sum(float(r['signal'])>=.65 for r in events)*1.5)))
        for variant,ev in [('original',events),('matched_signal',matched)]:
            eid=[r['event_id'] for r in ev]
            ef=c38.extract(ev,90)
            scores=detector.predict_proba(np.asarray([ef[i] for i in eid]))[:,1]
            y=np.asarray([byid[i] for i in eid])
            auroc=float(roc_auc_score(y,scores));ap=float(average_precision_score(y,scores))
            # Paired, fixed alert count: do NOT recalculate the budget from counterfactual signal.
            chosen={i for i,_ in sorted(zip(eid,scores),key=lambda z:(-float(z[1]),z[0]))[:original_budget]}
            rows=c58.top_rows(ev,chosen,new,90)
            if not rows:raise ValueError('No candidates for '+str((seed,variant)))
            gp=np.asarray(ng['model'].predict_proba(np.asarray([r['features'] for r in rows]))[:,1],dtype=float)
            cnt=np.asarray([int(round(np.expm1(r['features'][4]))) for r in rows],dtype=int)
            if np.any(cnt<1):raise ValueError('Invalid candidate count')
            dense=cnt>=16
            correct=np.asarray([(r['parent'],r['child']) in truth for r in rows],dtype=bool)
            for name,(s,d) in policies.items():
                keep=gp>=np.where(dense,d,s)
                tp=int(np.count_nonzero(keep&correct));fp=int(np.count_nonzero(keep&~correct))
                fn=len(truth)-tp
                output.append({'seed':seed,'variant':variant,'policy':name,'events':len(ev),'truth_edges':len(truth),
                               'alert_budget':original_budget,'candidate_children':len(rows),
                               'detector_auroc':round(auroc,8),'detector_ap':round(ap,8),
                               'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),8) if tp+fp else 0.,
                               'recall':round(tp/len(truth),8) if truth else 0.,
                               'f1':round(2*tp/(len(truth)+tp+fp),8) if len(truth)+tp+fp else 0.})
            print(f'seed={seed} {variant}: detector AUROC={auroc:.4f} AP={ap:.4f} alerts={original_budget} candidates={len(rows)}',flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'paired_metrics.csv',output)
    summary=[]
    for variant in ['original','matched_signal']:
        for name in policies:
            subset=[r for r in output if r['variant']==variant and r['policy']==name]
            tp=sum(r['tp'] for r in subset);fp=sum(r['fp'] for r in subset);n=sum(r['truth_edges'] for r in subset)
            summary.append({'variant':variant,'policy':name,'seeds':len(subset),
                            'mean_detector_auroc':round(float(np.mean([r['detector_auroc'] for r in subset])),6),
                            'mean_detector_ap':round(float(np.mean([r['detector_ap'] for r in subset])),6),
                            'mean_reconstruction_f1':round(float(np.mean([r['f1'] for r in subset])),6),
                            'pooled_reconstruction_f1':round(2*tp/(n+tp+fp),6),
                            'total_tp':tp,'total_fp':fp,'total_fn':n-tp})
    write(a.output/'summary.csv',summary)
    (a.output/'audit.json').write_text(json.dumps({'experiment':'E5-C70C','seeds':a.seeds,'control':'independent benign-marginal signal draws for all events','alert_budget':'fixed to original per-seed E5-C69 formula for both variants','train':'original E5-C15 train labels; frozen models and thresholds','note':'Synthetic generator robustness check; original seed set is reused from E5-C70B, not an independent confirmatory test','model_sha256':{'ranker':sha(a.retrained),'gate':sha(a.compatible_gate)}},indent=2),encoding='utf-8')
    print('\nE5-C70C SUMMARY')
    for row in summary:print(row)
    print('Saved:',a.output.resolve())
if __name__=='__main__':main()
