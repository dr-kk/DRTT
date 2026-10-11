"""E5-C84: frozen inference invariance to evaluation-label permutation.
Run from D:\\DRTT\\E5_ClosedLoop. Original telemetry only; no model training.
This is a dynamic invariance check, not a complete information-flow audit.
"""
import csv, hashlib, json, pickle, random, sys
from pathlib import Path
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76

OUT=ROOT/'results/causal/e5c84_label_permutation_audit'
SEEDS=list(range(8401,8406))
ALERTS=1800; EDGES=400; W=90

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def signature(value):return sha_bytes(pickle.dumps(value,protocol=4))
def dump(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def truth_edges(gt):
    return {(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
def permuted_truth(gt,seed):
    # Shuffle predecessor IDs among true-edge children. Labels are evaluation-only.
    pairs=sorted(truth_edges(gt))
    parents=[p for p,c in pairs]
    rng=random.Random(840000+seed)
    rng.shuffle(parents)
    return set(zip(parents,[c for p,c in pairs]))
def inference(events,detector,ranker,gate):
    ids=[r['event_id'] for r in events]
    feats=c38.extract(events,W)
    detector_scores=detector.predict_proba(np.asarray([feats[i] for i in ids],dtype=float))[:,1]
    order=sorted(zip(ids,map(float,detector_scores)),key=lambda z:(-z[1],z[0]))
    chosen={eid for eid,p in order[:ALERTS]}
    rr=c58.top_rows(events,chosen,ranker,W)
    gate_scores=gate['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1] if rr else np.asarray([])
    predicted=e75.top_budget(rr,gate_scores,EDGES) if rr else set()
    # Signatures use exact float64 bit patterns, sorted deterministic keys.
    return dict(alerts=tuple(sorted(chosen)),candidates=tuple(sorted((r['parent'],r['child']) for r in rr)),
        detector_scores=tuple(sorted((eid,float(p).hex()) for eid,p in zip(ids,detector_scores))),
        gate_scores=tuple(sorted((r['parent'],r['child'],float(p).hex()) for r,p in zip(rr,gate_scores))),
        predicted=tuple(sorted(predicted)))
def metrics(pred,truth):
    p=set(pred);tp=len(p&truth)
    return {'truth':len(truth),'predicted':len(p),'tp':tp,'fp':len(p)-tp,'fn':len(truth)-tp,
        'precision':tp/len(p) if p else 0.,'recall':tp/len(truth) if truth else 0.,
        'f1':2*tp/(len(p)+len(truth)) if len(p)+len(truth) else 0.}
def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
    paths={'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
        'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
        'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
        'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}
    for name,p in paths.items():
        if not p.is_file():raise FileNotFoundError(f'{name}: {p}')
    models={name:e75.load(p) for name,p in paths.items()}
    if models['gate']['features']!=c58.GATE_FEATURES or models['new_gate']['features']!=c58.GATE_FEATURES:
        raise RuntimeError('Gate feature mismatch')
    detectors=e76.fit_models();detector=detectors['OriginalOnly']
    rows=[]; all_invariant=True
    for seed in SEEDS:
        events,gt=gen.generate(seed,100,2400,10000)
        truth=truth_edges(gt); shuffled=permuted_truth(gt,seed)
        for name,rkey,gkey in [('Frozen','frozen','gate'),('HardNegative','hard','new_gate')]:
            before=inference(events,detector,models[rkey],models[gkey])
            # The only intervention is changing the labels supplied to the evaluator.
            # Re-run the inference function without supplying either label set.
            after=inference(events,detector,models[rkey],models[gkey])
            invariants={k:before[k]==after[k] for k in before}
            all_invariant &= all(invariants.values())
            m0=metrics(before['predicted'],truth);m1=metrics(after['predicted'],shuffled)
            row=dict(seed=seed,method=name,truth_changed=(truth!=shuffled),
                **{'same_'+k:int(v) for k,v in invariants.items()},
                baseline_tp=m0['tp'],permuted_tp=m1['tp'],baseline_f1=m0['f1'],permuted_f1=m1['f1'],
                prediction_sha256=signature(before['predicted']),candidate_sha256=signature(before['candidates']))
            rows.append(row)
            print(f"seed={seed} {name} invariant={all(invariants.values())} TP {m0['tp']} -> {m1['tp']}",flush=True)
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'test_seed_metrics.csv',rows)
    audit={'experiment':'E5-C84','test_seeds':SEEDS,'alerts':ALERTS,'edge_budget':EDGES,
        'all_predictions_and_scores_invariant':bool(all_invariant),
        'method':'Frozen models; identical original telemetry; shuffled predecessor IDs in evaluation truth; exact repeated inference equality',
        'model_sha256':{k:sha_bytes(p.read_bytes()) for k,p in paths.items()},
        'limitations':['Dynamic label-permutation check only: cannot rule out training-time leakage, generator-induced proxies, or direct truth access in uninspected dependencies',
        'Ground truth is not an argument of inference(), so this checks repeatability and separation from evaluation labels, not a mutation of any hidden global label store',
        'Matched-signal counterfactual intentionally excluded because its generation uses benign ground-truth labels',
        'Synthetic generator only; fixed 1800 alerts and 400 edges; five new seeds']}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('E5-C84', 'PASS (limited dynamic check)' if all_invariant else 'FAIL', '\nSaved:',OUT)
    if not all_invariant:raise SystemExit(2)
if __name__=='__main__':main()
