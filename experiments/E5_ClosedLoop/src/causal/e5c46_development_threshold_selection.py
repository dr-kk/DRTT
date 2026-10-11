"""E5-C46: development-only candidate-validity threshold selection.

Uses E5-C17 calibration labels solely for development selection; do not report
these results as independent test performance. Does not modify frozen artifacts.
"""
import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--min-tp-retention',type=float,default=.90)
    a=ap.parse_args()
    if not 0<a.min_tp_retention<=1:ap.error('Retention must be in (0,1]')
    if a.train.resolve()==a.calibration.resolve():ap.error('Train/calibration must differ')
    if a.output.exists() and any(a.output.iterdir()):ap.error('Output directory must be empty')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c45_end_to_end_candidate_filter as c45
    import e5c15_reranker as base
    with a.models.open('rb') as f:bundle=pickle.load(f)
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if bundle['ranker_sha256']!=sha(a.ranker) or gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Ranker hash mismatch')
    if bundle['features']!=c41.FEATURES or gate['features']!=c38.GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature schema mismatch')
    if int(gate['window'])!=90 or int(ranker['window'])!=90 or float(ranker['threshold'])!=.65:raise ValueError('Frozen settings mismatch')
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    train_feat=c38.extract(train,90)
    train_ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([train_feat[i] for i in train_ids]),np.asarray([labels[i] for i in train_ids]))
    events=c38.read(a.calibration/'telemetry.csv')
    gt=c38.read(a.calibration/'ground_truth.csv')
    truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r['predecessor_event_id']}
    ids=[r['event_id'] for r in events]
    feat=c38.extract(events,90)
    probabilities=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
    alert_count=sum(float(r['signal'])>=.65 for r in events)
    selected={i for i,_ in sorted(zip(ids,probabilities),key=lambda x:(-float(x[1]),x[0]))[:alert_count]}
    records=c45.candidates_with_features(events,selected,ranker,90)
    expected,audit=c38.top_predictions(events,selected,ranker,gate,base,90)
    if len(records)!=audit['candidate_children']:raise AssertionError('Candidate count mismatch')
    gate_probs=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records]))[:,1]
    accepted=np.asarray(gate_probs>=gate['cutoff'])
    baseline={(r['parent'],r['child']) for r,yes in zip(records,accepted) if yes}
    if baseline!=set(expected):raise AssertionError('Baseline mismatch')
    base_tp=len(baseline&truth);base_fp=len(baseline-truth)
    if base_tp==0:raise ValueError('No correct baseline edges; cannot select retention threshold')
    validity=bundle['models']['unweighted'].predict_proba(np.asarray([r['valid_features'] for r in records]))[:,1]
    # Exhaustive observed-score thresholds; no arbitrary threshold grid artifacts.
    # A threshold of 0 means keep every baseline edge, regardless of probability.
    options=sorted({0.0,*[float(v) for v,yes in zip(validity,accepted) if yes]})
    rows=[]
    for threshold in options:
        edges={(r['parent'],r['child']) for r,g,v in zip(records,accepted,validity) if g and v>=threshold}
        tp=len(edges&truth);fp=len(edges)-tp
        prec=tp/len(edges) if edges else 0.0
        rec=tp/len(truth) if truth else 0.0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
        rows.append({'threshold':threshold,'tp':tp,'fp':fp,'predicted':len(edges),'tp_retention':tp/base_tp,
                     'fp_reduction':(base_fp-fp)/base_fp if base_fp else 0.0,
                     'precision':prec,'recall':rec,'f1':f1,'feasible':int(tp/base_tp>=a.min_tp_retention)})
    feasible=[r for r in rows if r['feasible']]
    # Choose lowest FP among thresholds satisfying the retention constraint;
    # break ties by higher TP and then lower threshold.
    chosen=min(feasible,key=lambda r:(r['fp'],-r['tp'],r['threshold']))
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'threshold_sweep.csv',rows)
    report={'experiment':'E5-C46','scope':'development-only E5-C17 threshold selection; NOT independent validation',
      'selected_model':'unweighted','selected_threshold':chosen['threshold'],
      'selection_rule':'minimize false accepted edges subject to >=90% (configurable) of baseline true edges; tie break higher TP, then lower threshold',
      'min_tp_retention':a.min_tp_retention,
      'baseline':{'tp':base_tp,'fp':base_fp,'predicted':len(baseline),'truth_edges':len(truth),'candidate_children':len(records),'alerts':alert_count},
      'selected_development_metrics':chosen,'thresholds_examined':len(rows),
      'sha256':{'models':sha(a.models),'ranker':sha(a.ranker),'gate':sha(a.gate),
                'train_telemetry':sha(a.train/'telemetry.csv'),'calibration_telemetry':sha(a.calibration/'telemetry.csv')},
      'limitations':['E5-C17 has been used in prior development; selected metrics are optimistic',
       'No evaluation on E5-C44 or E5-C45 seeds for selection',
       'Validity model was trained on signal-selected candidates, but applied to context-selected candidates',
       'Same synthetic generator, no operational validation',
       'Do not tune this threshold after inspecting a subsequent held-out test']}
    (a.output/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C46 | DEVELOPMENT-ONLY OPERATING POINT SELECTION')
    print(f'Baseline TP/FP={base_tp}/{base_fp}; truth edges={len(truth)}; candidate children={len(records)}')
    print(f'Feasible thresholds={len(feasible)}/{len(rows)}; required TP retention={a.min_tp_retention:.1%}')
    print(f'Selected threshold={chosen["threshold"]:.12g}: TP/FP={chosen["tp"]}/{chosen["fp"]}; retention={chosen["tp_retention"]:.2%}; FP reduction={chosen["fp_reduction"]:.2%}; F1={chosen["f1"]:.4f}')
    print('WARNING: E5-C17 reused for development; no independent performance claim.')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
