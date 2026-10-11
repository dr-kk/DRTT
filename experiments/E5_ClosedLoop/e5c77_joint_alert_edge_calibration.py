"""E5-C77: validation-only joint alert and edge-budget selection.
Run in D:\\DRTT\\E5_ClosedLoop. Does not modify earlier experiments.
Trains C76 detectors on their original training seeds, reuses frozen rankers/gates.
Selection: 7701-7703; untouched test: 7704-7708.
Synthetic-only; signal matching is a label-informed evaluation perturbation.
"""
import csv, hashlib, json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c15_reranker as base
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76
import e5c5_overlapping_generator as gen

OUT=ROOT/'results/causal/e5c77_joint_alert_edge_calibration'
VAL=[7701,7702,7703]
TEST=[7704,7705,7706,7707,7708]
ALERTS=[500,900,1300,1800]
EDGES=[200,400,600]
VARIANTS=['original','matched_signal']
W=90

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:
        key=(r['split'],r['detector'],r['ranker'],r['variant'],r['alerts'],r['edge_budget'])
        grouped[key].append(r)
    out=[]
    for key,rr in grouped.items():
        tp=sum(x['tp'] for x in rr);pred=sum(x['predicted'] for x in rr);truth=sum(x['truth'] for x in rr)
        out.append(dict(zip(('split','detector','ranker','variant','alerts','edge_budget'),key),seeds=len(rr),truth=truth,predicted=pred,
                        tp=tp,fp=pred-tp,fn=truth-tp,precision=tp/pred if pred else 0.,recall=tp/truth if truth else 0.,
                        pooled_f1=e75.f1(tp,pred,truth),mean_alert_edge_ceiling=float(np.mean([x['alert_edge_ceiling'] for x in rr]))))
    return out

def evaluate(seeds,split,detectors,rankers,gates,configs=None):
    results=[]
    for seed in seeds:
        events,gt=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
        labels={r['event_id']:int(r['malicious']) for r in gt}
        benign=[r['signal'] for r in events if labels[r['event_id']]==0]
        import random
        rng=random.Random(170000+seed)
        matched=[dict(r,signal=rng.choice(benign)) for r in events]
        for variant,ev in [('original',events),('matched_signal',matched)]:
            ids=[r['event_id'] for r in ev]
            features=c38.extract(ev,W)
            X=np.asarray([features[i] for i in ids],dtype=float)
            for dname,detector in detectors.items():
                probabilities=detector.predict_proba(X)[:,1]
                sorted_ids=[i for i,_ in sorted(zip(ids,probabilities),key=lambda z:(-float(z[1]),z[0]))]
                for alerts in ALERTS:
                    selected=set(sorted_ids[:alerts])
                    ceiling=sum((parent in selected and child in selected) for parent,child in truth)
                    for rname,ranker in rankers.items():
                        if configs is not None and not any(c['detector']==dname and c['ranker']==rname and c['alerts']==alerts for c in configs):
                            continue
                        rr=c58.top_rows(ev,selected,ranker,W)
                        if rr:
                            scores=gates[rname]['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1]
                            order=sorted(range(len(rr)),key=lambda i:(-float(scores[i]),rr[i]['child'],rr[i]['parent']))
                            ranked=[(rr[i]['parent'],rr[i]['child']) for i in order]
                        else:ranked=[]
                        budgets=EDGES if configs is None else sorted({c['edge_budget'] for c in configs if c['detector']==dname and c['ranker']==rname and c['alerts']==alerts})
                        for budget in budgets:
                            pred=set(ranked[:budget]);tp=len(pred&truth)
                            results.append(dict(split=split,seed=seed,variant=variant,detector=dname,ranker=rname,alerts=alerts,
                                                edge_budget=budget,truth=len(truth),alert_edge_ceiling=ceiling/len(truth),
                                                candidates=len(rr),predicted=len(pred),tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,
                                                f1=e75.f1(tp,len(pred),len(truth))))
        print(f'{split} seed={seed} complete',flush=True)
    return results

def select(validation_summary,objective):
    grouped=defaultdict(dict)
    for r in validation_summary:
        k=(r['detector'],r['ranker'],r['alerts'],r['edge_budget'])
        grouped[k][r['variant']]=r['pooled_f1']
    candidates=[]
    for k,v in grouped.items():
        if len(v)!=2:continue
        orig=v['original'];matched=v['matched_signal']
        score=(orig+matched)/2 if objective=='balanced' else min(orig,matched)
        candidates.append(dict(detector=k[0],ranker=k[1],alerts=k[2],edge_budget=k[3],validation_original_f1=orig,
                               validation_matched_f1=matched,validation_score=score,objective=objective))
    return sorted(candidates,key=lambda r:(-r['validation_score'],-r['validation_original_f1'],r['edge_budget'],r['alerts'],r['detector'],r['ranker']))[0]

def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
    paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
           'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
           'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
           'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
           'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}
    for k,p in paths.items():
        if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
    old=e75.load(paths['old']);frozen=e75.load(paths['frozen']);hard=e75.load(paths['hard'])
    gate=e75.load(paths['gate']);new_gate=e75.load(paths['new_gate'])
    if old['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1] or hard['features']!=base.FEATURES[:-1]:
        raise RuntimeError('Ranker feature mismatch')
    if gate['features']!=c58.GATE_FEATURES or new_gate['features']!=c58.GATE_FEATURES:
        raise RuntimeError('Gate feature mismatch')
    if frozen.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['frozen']):
        raise RuntimeError('Frozen model lineage mismatch')
    audit74=json.loads((paths['hard'].parent/'audit.json').read_text(encoding='utf-8'))
    if hard.get('selection')!=audit74.get('selected_config'):raise RuntimeError('Hard-negative provenance mismatch')
    detectors=e76.fit_models()
    rankers={'Frozen':frozen,'HardNegative':hard}
    gates={'Frozen':gate,'HardNegative':new_gate}
    val=evaluate(VAL,'validation',detectors,rankers,gates)
    val_summary=summarize(val)
    selections=[select(val_summary,'balanced'),select(val_summary,'worst_case')]
    # Include an independently specified control: OriginalOnly + Frozen, 900 alerts, 400 edges.
    control=dict(detector='OriginalOnly',ranker='Frozen',alerts=900,edge_budget=400,objective='prespecified_control')
    configs=selections+[control]
    print('Validation selections:')
    for c in configs:print(c,flush=True)
    test=evaluate(TEST,'test',detectors,rankers,gates,configs=configs)
    test_summary=summarize(test)
    selected=[]
    for config in configs:
        for r in test_summary:
            if all(r[k]==config[k] for k in ('detector','ranker','alerts','edge_budget')):
                selected.append(dict(objective=config['objective'],**r))
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'validation_seed_metrics.csv',val)
    write(OUT/'validation_summary.csv',val_summary)
    write(OUT/'selected_configurations.csv',configs)
    write(OUT/'test_seed_metrics.csv',test)
    write(OUT/'test_summary.csv',selected)
    (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C77',training_seeds=e76.TRAIN,validation_seeds=VAL,test_seeds=TEST,
        alert_grid=ALERTS,edge_budget_grid=EDGES,selection_criteria=['mean_original_matched_F1','minimum_original_matched_F1'],
        selected_configurations=configs,input_sha256={k:sha(p) for k,p in paths.items()},
        controls='OriginalOnly + Frozen, 900 alerts, 400 edges, prespecified',
        caveats=['Synthetic-only','Matched signal is label-informed perturbation; labels not passed to inference',
                 'All gate models frozen; no new gate training','Selection uses validation labels, never test labels',
                 'Fixed absolute budgets across conditions, not test-derived',
                 'Only selected configurations are evaluated on test seeds',
                 'Budgeted top-k confidence is not probability calibration',
                 'No paired confidence intervals or host randomization']),indent=2),encoding='utf-8')
    print('\nE5-C77 TEST SUMMARY')
    for r in selected:print(r)
    print('Saved:',OUT)

if __name__=='__main__':main()
