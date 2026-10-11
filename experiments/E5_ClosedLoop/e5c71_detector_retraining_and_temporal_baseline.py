"""E5-C71: paired detector retraining and simple temporal-host baseline.
Run from D:\\DRTT\\E5_ClosedLoop. Never overwrites existing results.
NOTE: reranker and confidence gate remain frozen; this isolates detector retraining.
"""
import argparse, csv, hashlib, json, pickle, random, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def matched(events,labels,seed):
    benign=[r['signal'] for r in events if not labels[r['event_id']]]
    rng=random.Random(170000+seed)
    return [dict(r,signal=rng.choice(benign)) for r in events]

def train_detector(events,labels,extract):
    features=extract(events,90)
    ids=[r['event_id'] for r in events]
    X=np.asarray([features[i] for i in ids],dtype=float)
    y=np.asarray([labels[i] for i in ids],dtype=int)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    model.fit(X,y)
    return model

def score_detector(model,events,labels,extract):
    ids=[r['event_id'] for r in events]
    features=extract(events,90)
    X=np.asarray([features[i] for i in ids],dtype=float)
    y=np.asarray([labels[i] for i in ids],dtype=int)
    scores=model.predict_proba(X)[:,1]
    return ids,scores,float(roc_auc_score(y,scores)),float(average_precision_score(y,scores))

def temporal_host_baseline(events,chosen,window=90):
    """Most-recent eligible parent, without a learned ranker or gate.
    At most one edge per selected child; host-continuity eligibility identical
    to the frozen pipeline. No ground-truth information is used.
    """
    selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
    edges=[]
    for j,child in enumerate(selected):
        ct=int(child['timestamp'])
        for parent in reversed(selected[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if (parent['destination_host']==child['source_host'] or
                parent['source_host']==child['source_host'] or
                parent['destination_host']==child['destination_host']):
                edges.append((parent['event_id'],child['event_id']))
                break
    return edges

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--retrained',type=Path,default=Path('results/causal/e5c56/reranker_no_gap_rank.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--compatible-gate',type=Path,default=Path('results/causal/e5c58/compatible_gate.pkl'))
    p.add_argument('--safety-policy',type=Path,default=Path('results/causal/e5c64/constrained_gate_policy.json'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c71_detector_retraining'))
    p.add_argument('--seeds',type=int,nargs='+',default=[7101,7102,7103,7104,7105])
    a=p.parse_args()
    if a.output.exists():p.error('Output directory already exists; refusing overwrite')
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
    if old['features']!=base.FEATURES or new['features']!=base.FEATURES[:-1]:raise ValueError('Ranker schema mismatch')
    if og['features']!=c58.GATE_FEATURES or ng['features']!=c58.GATE_FEATURES:raise ValueError('Gate schema mismatch')
    if new.get('baseline_sha256')!=digest(a.ranker) or og.get('e5c15_sha256')!=digest(a.ranker):raise ValueError('Ranker lineage mismatch')
    if ng.get('e5c56_sha256')!=digest(a.retrained) or ng.get('baseline_e5c15_sha256')!=digest(a.ranker):raise ValueError('Gate lineage mismatch')
    if abs(float(ng['budget'])-1.5)>1e-9:raise ValueError('Alert budget mismatch')
    safety=json.loads(a.safety_policy.read_text(encoding='utf-8'))
    if safety.get('experiment')!='E5-C64' or safety.get('model_sha256',{}).get('proposed_ranker')!=digest(a.retrained) or safety.get('model_sha256',{}).get('compatible_gate')!=digest(a.compatible_gate):raise ValueError('Policy lineage mismatch')
    if int(safety['density_boundary'])!=16 or abs(float(safety['original_global_cutoff'])-float(ng['cutoff']))>1e-9:raise ValueError('Policy mismatch')
    policies={'Global':(float(ng['cutoff']),float(ng['cutoff'])),
              'E5-C64':(float(safety['sparse_cutoff']),float(safety['dense_cutoff'])),
              'E5-C66A':(.725,.800)}
    train=c58.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
    models={'original_train':train_detector(train,labels,c38.extract),
            'matched_train':train_detector(matched(train,labels,7100),labels,c38.extract)}
    rows=[]
    for seed in a.seeds:
        events,ground=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
        labels_test={r['event_id']:int(r['malicious']) for r in ground}
        variants={'original_test':events,'matched_test':matched(events,labels_test,seed)}
        # Both variants use original raw-signal alert budget (as in E5-C70C).
        budget=min(len(events),int(round(sum(float(r['signal'])>=.65 for r in events)*1.5)))
        for train_name,model in models.items():
            for test_name,ev in variants.items():
                ids,scores,auc,ap=score_detector(model,ev,labels_test,c38.extract)
                chosen={i for i,_ in sorted(zip(ids,scores),key=lambda x:(-float(x[1]),x[0]))[:budget]}
                candidates=c58.top_rows(ev,chosen,new,90)
                if not candidates:raise RuntimeError('No candidates')
                gp=np.asarray(ng['model'].predict_proba(np.asarray([r['features'] for r in candidates]))[:,1],dtype=float)
                density=np.asarray([int(round(np.expm1(r['features'][4]))) for r in candidates],dtype=int)
                if np.any(density<1):raise ValueError('Invalid candidate count')
                correct=np.asarray([(r['parent'],r['child']) in truth for r in candidates],dtype=bool)
                for policy,(s,d) in policies.items():
                    keep=gp>=np.where(density>=16,d,s)
                    tp=int(np.count_nonzero(keep&correct));fp=int(np.count_nonzero(keep&~correct))
                    rows.append(dict(seed=seed,train=train_name,test=test_name,policy=policy,events=len(ev),
                                     truth_edges=len(truth),alert_budget=budget,candidate_children=len(candidates),
                                     detector_auroc=round(auc,8),detector_ap=round(ap,8),
                                     tp=tp,fp=fp,fn=len(truth)-tp,
                                     precision=round(tp/(tp+fp),8) if tp+fp else 0.,
                                     recall=round(tp/len(truth),8),f1=round(2*tp/(len(truth)+tp+fp),8)))
                edges=temporal_host_baseline(ev,chosen)
                tp=sum(edge in truth for edge in edges);fp=len(edges)-tp
                rows.append(dict(seed=seed,train=train_name,test=test_name,policy='TemporalHost',events=len(ev),
                                 truth_edges=len(truth),alert_budget=budget,candidate_children=len(edges),
                                 detector_auroc=round(auc,8),detector_ap=round(ap,8),
                                 tp=tp,fp=fp,fn=len(truth)-tp,
                                 precision=round(tp/(tp+fp),8) if tp+fp else 0.,
                                 recall=round(tp/len(truth),8),f1=round(2*tp/(len(truth)+tp+fp),8)))
                print(f'{seed} {train_name} -> {test_name}: AUROC={auc:.4f} AP={ap:.4f} alerts={budget}',flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'seed_metrics.csv',rows)
    summary=[]
    for train_name in models:
        for test_name in ['original_test','matched_test']:
            for policy in [*policies,'TemporalHost']:
                subset=[r for r in rows if r['train']==train_name and r['test']==test_name and r['policy']==policy]
                tp=sum(r['tp'] for r in subset);fp=sum(r['fp'] for r in subset);n=sum(r['truth_edges'] for r in subset)
                summary.append(dict(train=train_name,test=test_name,policy=policy,seeds=len(subset),
                                    mean_auroc=round(float(np.mean([r['detector_auroc'] for r in subset])),6),
                                    mean_ap=round(float(np.mean([r['detector_ap'] for r in subset])),6),
                                    mean_f1=round(float(np.mean([r['f1'] for r in subset])),6),
                                    pooled_f1=round(2*tp/(n+tp+fp),6),tp=tp,fp=fp,fn=n-tp))
    write(a.output/'summary.csv',summary)
    (a.output/'audit.json').write_text(json.dumps(dict(experiment='E5-C71',seeds=a.seeds,
        detector_training='original vs matched benign-marginal signal; only detector retrained',
        reconstruction='E5-C56 ranker and E5-C58 gate frozen; E5-C64/E5-C66A thresholds frozen',
        alert_budget='paired original-signal E5-C69 formula, fixed across train/test variants',
        temporal_host='most recent selected eligible parent within 90 units; no learned ranker/gate',
        limitation='Signal-matched detector retraining is not full-pipeline retraining; no independent real-world validation',
        model_sha256={'ranker':digest(a.retrained),'gate':digest(a.compatible_gate)}),indent=2),encoding='utf-8')
    print('\nE5-C71 SUMMARY')
    for r in summary:print(r)
    print('Saved:',a.output.resolve())
if __name__=='__main__':main()
