"""E5-C76: signal-robust detector, independent seed evaluation.
Run from D:\\DRTT\\E5_ClosedLoop with e5c75_calibrated_end_to_end.py alongside.
No test labels used for fitting, model selection, alert selection or edge budgeting.
Original and matched telemetry are synthetic; results do not imply real-world validity.
"""
import csv, hashlib, json, pickle, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base
import e5c75_calibrated_end_to_end as e75

OUT=ROOT/'results/causal/e5c76_signal_robust_detector'
TRAIN=list(range(7601,7611)); VAL=list(range(7611,7614)); TEST=list(range(7614,7619))
W=90

class Detector:
    def __init__(self, model, drop_signal=False):
        self.model=model; self.drop_signal=drop_signal
    def predict_proba(self,X):
        X=np.asarray(X,dtype=float)
        return self.model.predict_proba(X[:,1:] if self.drop_signal else X)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def variants(seed):
    events,gt=gen.generate(seed,100,2400,10000)
    labels={r['event_id']:int(r['malicious']) for r in gt}
    benign=[r['signal'] for r in events if labels[r['event_id']]==0]
    rng=random.Random(170000+seed)
    matched=[dict(r,signal=rng.choice(benign)) for r in events]
    return labels,{'original':events,'matched_signal':matched}

def xy(events,labels):
    feat=c38.extract(events,W)
    ids=[r['event_id'] for r in events]
    return np.asarray([feat[i] for i in ids],dtype=float),np.asarray([labels[i] for i in ids],dtype=int)

def fit_models():
    parts={'original':[],'matched_signal':[]};ys=[]
    for seed in TRAIN:
        labels,ev=variants(seed)
        for variant in parts:
            X,y=xy(ev[variant],labels);parts[variant].append(X)
        ys.append(y)
        print(f'train seed={seed} events={len(y)}',flush=True)
    y=np.concatenate(ys)
    xo=np.vstack(parts['original']);xm=np.vstack(parts['matched_signal'])
    def fit(X,y):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=0))
        m.fit(X,y);return m
    ya=np.concatenate([y,y]);xa=np.vstack([xo,xm])
    return {'OriginalOnly':Detector(fit(xo,y)),
            'Augmented':Detector(fit(xa,ya)),
            'NoSignal':Detector(fit(xa[:,1:],ya),True)}

def detector_metrics(seed,model,labels,ev):
    out=[]
    for variant,events in ev.items():
        X,y=xy(events,labels)
        p=model.predict_proba(X)[:,1]
        out.append(dict(seed=seed,variant=variant,auroc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p))))
    return out

def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
    paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
           'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
           'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
           'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
           'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl',
           'e75_audit':ROOT/'results/causal/e5c75_calibrated_end_to_end/audit.json'}
    for k,p in paths.items():
        if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
    loaded={k:e75.load(p) for k,p in paths.items() if k not in ('e75_audit',)}
    frozen,hard,gate,new_gate=loaded['frozen'],loaded['hard'],loaded['gate'],loaded['new_gate']
    if loaded['old']['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1] or hard['features']!=base.FEATURES[:-1]:
        raise RuntimeError('Reranker schema mismatch')
    if gate['features']!=c58.GATE_FEATURES or new_gate['features']!=c58.GATE_FEATURES:
        raise RuntimeError('Gate schema mismatch')
    if frozen.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['frozen']):
        raise RuntimeError('Frozen lineage mismatch')
    audit75=json.loads(paths['e75_audit'].read_text(encoding='utf-8'))
    if audit75.get('test_seeds')!=list(range(7509,7514)):
        raise RuntimeError('E5-C75 audit mismatch')
    models=fit_models()
    validation=[]
    for seed in VAL:
        labels,ev=variants(seed)
        for name,model in models.items():
            for r in detector_metrics(seed,model,labels,ev):validation.append(dict(model=name,**r))
        print(f'validation seed={seed}',flush=True)
    val_groups=defaultdict(list)
    for r in validation:val_groups[r['model']].append(r)
    val_summary=[dict(model=k,mean_ap=float(np.mean([r['ap'] for r in v])),mean_auroc=float(np.mean([r['auroc'] for r in v])),
                      matched_ap=float(np.mean([r['ap'] for r in v if r['variant']=='matched_signal']))) for k,v in val_groups.items()]
    winner=sorted(val_summary,key=lambda r:(-r['mean_ap'],-r['matched_ap'],r['model']))[0]['model']
    print('Validation selected detector:',winner,flush=True)
    # Evaluate every detector configuration, not only the validation winner.
    test_detector=[];test_edges=[];test_ceiling=[]
    for seed in TEST:
        labels,ev=variants(seed)
        for name,model in models.items():
            for r in detector_metrics(seed,model,labels,ev):test_detector.append(dict(model=name,**r))
            truth,vmap,budget,n=e75.prepare(seed,model,frozen,hard,gate)
            for variant,(fr,nr) in vmap.items():
                gt={child:parent for parent,child in truth}
                for ranker,rr,gg in [('Frozen',fr,gate['model']),('HardNegative',nr,new_gate['model'])]:
                    X=np.asarray([r['features'] for r in rr],dtype=float)
                    probs=gg.predict_proba(X)[:,1] if len(rr) else np.array([])
                    pred=e75.top_budget(rr,probs,budget)
                    tp=len(pred&truth)
                    test_edges.append(dict(seed=seed,detector=name,variant=variant,ranker=ranker,
                                           truth=len(truth),alerts=n,predicted=len(pred),tp=tp,fp=len(pred)-tp,
                                           fn=len(truth)-tp,f1=e75.f1(tp,len(pred),len(truth))))
                    reachable_children=sum(1 for r in rr if (gt.get(r['child'])==r['parent']))
                    # Each rr contains one selected parent per child; this is top-1 correctness, NOT candidate ceiling.
                    test_ceiling.append(dict(seed=seed,detector=name,variant=variant,ranker=ranker,
                                             selected_top1_correct=reachable_children))
        print(f'test seed={seed}',flush=True)
    groups=defaultdict(list)
    for r in test_edges:groups[(r['detector'],r['variant'],r['ranker'])].append(r)
    summary=[]
    for (det,variant,ranker),rr in groups.items():
        tp=sum(x['tp'] for x in rr);pred=sum(x['predicted'] for x in rr);truth=sum(x['truth'] for x in rr)
        summary.append(dict(detector=det,variant=variant,ranker=ranker,seeds=len(rr),truth=truth,predicted=pred,
                            tp=tp,fp=pred-tp,fn=truth-tp,precision=tp/pred if pred else 0.,
                            recall=tp/truth if truth else 0.,pooled_f1=e75.f1(tp,pred,truth)))
    detector_summary=[]
    for name in models:
        for variant in ('original','matched_signal'):
            rr=[r for r in test_detector if r['model']==name and r['variant']==variant]
            detector_summary.append(dict(detector=name,variant=variant,seeds=len(rr),mean_auroc=float(np.mean([r['auroc'] for r in rr])),mean_ap=float(np.mean([r['ap'] for r in rr]))))
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'validation_detector_metrics.csv',validation)
    write(OUT/'validation_summary.csv',val_summary)
    write(OUT/'test_detector_seed_metrics.csv',test_detector)
    write(OUT/'test_detector_summary.csv',detector_summary)
    write(OUT/'test_edge_seed_metrics.csv',test_edges)
    write(OUT/'test_edge_summary.csv',summary)
    write(OUT/'test_top1_diagnostics.csv',test_ceiling)
    with (OUT/'selected_detector.pkl').open('wb') as f:pickle.dump(models[winner],f)
    (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C76',train_seeds=TRAIN,validation_seeds=VAL,test_seeds=TEST,
        validation_winner=winner,validation_criterion='Mean detector AP over both original and matched validation conditions',
        detector_variants=list(models),feature_schema='c38 four features; NoSignal excludes signal column',
        gate_policy='Frozen gate and E5-C75 separately fitted hard-negative gate; no new gate tuning',
        edge_budget='E5-C75 per-seed original frozen cutoff count, shared across models and matched conditions',
        original_vs_matched='Signal values resampled from benign marginal independently; not a realistic network shift',
        limitations=['Synthetic-only','Same signal-dependent alert budget rule from E5-C75','No host-randomized condition',
                     'No gate recalibration for the new detectors','No statistical significance test',
                     'All model variants reported; selected detector based on validation only'],
        input_sha256={k:sha(p) for k,p in paths.items()}),indent=2),encoding='utf-8')
    print('\nE5-C76 DETECTOR SUMMARY')
    for r in detector_summary:print(r)
    print('\nE5-C76 EDGE SUMMARY')
    for r in summary:print(r)
    print('Saved:',OUT)
if __name__=='__main__':main()
