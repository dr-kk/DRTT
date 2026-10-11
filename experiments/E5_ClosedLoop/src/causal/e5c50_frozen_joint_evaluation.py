"""E5-C50: frozen joint selected-parent correctness model, synthetic seeds 5000-5004.
No threshold tuning. Existing E5-C31 gate vs E5-C49 joint model at 0.50.
"""
import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(y, prob):
    if len(set(y))!=2:raise ValueError('Both classes needed for ranking metrics')
    return {'n':len(y),'correct_top1':int(sum(y)),
            'prevalence':float(np.mean(y)),
            'auroc':float(roc_auc_score(y,prob)),
            'average_precision':float(average_precision_score(y,prob)),
            'brier':float(brier_score_loss(y,prob)),
            'log_loss':float(log_loss(y,prob,labels=[0,1])),
            'mean_probability':float(np.mean(prob))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--joint',type=Path,default=Path('results/causal/e5c49/joint_model.pkl'))
    ap.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,default=Path('results/causal/e5c50'))
    a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error('Output directory must be empty; previous results preserved')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.joint.open('rb') as f:joint=pickle.load(f)
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if joint['features']!=c41.FEATURES+c38.GATE_FEATURES:raise ValueError('Joint feature order differs')
    if len(joint['features'])!=29:raise ValueError('Expected 29 feature positions')
    if ranker['features']!=base.FEATURES or gate['features']!=c38.GATE_FEATURES:raise ValueError('Feature mismatch')
    if joint['sha256']['ranker']!=sha(a.ranker) or joint['sha256']['gate']!=sha(a.gate):raise ValueError('Frozen model hash mismatch')
    if gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Gate ranker hash mismatch')
    if joint['sha256']['train_telemetry']!=sha(a.train/'telemetry.csv'):raise ValueError('Training telemetry hash mismatch')
    if float(ranker['threshold'])!=.65 or int(ranker['window'])!=90 or int(gate['window'])!=90:raise ValueError('Unexpected frozen config')
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    feat=c38.extract(train,90);ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    per_seed=[];candidate_rows=[];manifest=[]
    for seed in range(5000,5005):
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        x=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([x[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
        records=c45.candidates_with_features(events,chosen,ranker,90)
        expected,audit=c38.top_predictions(events,chosen,ranker,gate,base,90)
        if len(records)!=audit['candidate_children']:raise AssertionError('Candidate count mismatch')
        if not records:raise AssertionError('Unexpected empty candidate set')
        G=np.asarray([r['gate_features'] for r in records],dtype=float)
        X=np.asarray([r['valid_features']+r['gate_features'] for r in records],dtype=float)
        if X.shape[1]!=29:raise AssertionError('Feature vector size mismatch')
        gp=gate['model'].predict_proba(G)[:,1]
        jp=joint['model'].predict_proba(X)[:,1]
        if not np.isfinite(gp).all() or not np.isfinite(jp).all():raise ValueError('Non-finite probabilities')
        baseline={(r['parent'],r['child']) for r,p in zip(records,gp) if p>=gate['cutoff']}
        if baseline!=set(expected):raise AssertionError(f'Seed {seed}: baseline differs from E5-C38')
        joint_edges={(r['parent'],r['child']) for r,p in zip(records,jp) if p>=.5}
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        y=np.asarray([int((r['parent'],r['child']) in truth) for r in records],dtype=int)
        for name,edges in [('context_frozen_gate',baseline),('context_joint_0_5',joint_edges)]:
            m=c45.evaluate(edges,truth,len(events))
            per_seed.append({'seed':seed,'pipeline':name,'alerts':n,'candidate_children':len(records),**m})
        for r,yi,g,j in zip(records,y,gp,jp):
            candidate_rows.append({'seed':seed,'parent':r['parent'],'child':r['child'],
                                   'correct_top1':int(yi),'gate_probability':float(g),
                                   'joint_probability':float(j),
                                   'gate_accepted':int(g>=gate['cutoff']),
                                   'joint_accepted':int(j>=.5)})
        bm=c45.evaluate(baseline,truth,len(events));jm=c45.evaluate(joint_edges,truth,len(events))
        manifest.append({'seed':seed,'events':len(events),'truth_edges':len(truth),'alerts':n,'candidate_children':len(records)})
        print(f'Seed {seed}: gate TP/FP={bm["tp"]}/{bm["fp"]}; joint@0.50 TP/FP={jm["tp"]}/{jm["fp"]}; delta F1={jm["f1"]-bm["f1"]:+.5f}')
    summaries={}
    for name in ('context_frozen_gate','context_joint_0_5'):
        rr=[r for r in per_seed if r['pipeline']==name]
        tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr)
        summaries[name]={'tp':tp,'fp':fp,'precision_pooled':tp/(tp+fp) if tp+fp else 0,
                         'recall_mean':float(np.mean([r['recall'] for r in rr])),
                         'f1_mean':float(np.mean([r['f1'] for r in rr])),
                         'false_edges_per_1000_events_mean':float(np.mean([r['false_edges_per_1000_events'] for r in rr]))}
    yy=np.asarray([r['correct_top1'] for r in candidate_rows],dtype=int)
    gg=np.asarray([r['gate_probability'] for r in candidate_rows],dtype=float)
    jj=np.asarray([r['joint_probability'] for r in candidate_rows],dtype=float)
    b=summaries['context_frozen_gate'];j=summaries['context_joint_0_5']
    report={'experiment':'E5-C50','scope':'frozen synthetic evaluation seeds 5000-5004; no test-based threshold tuning',
            'decision_thresholds':{'existing_gate':float(gate['cutoff']),'joint':0.5},
            'manifest':manifest,'summary':summaries,
            'candidate_correctness':{'existing_gate':metrics(yy,gg),'joint_logistic':metrics(yy,jj)},
            'paired_aggregate':{'delta_tp':j['tp']-b['tp'],'delta_fp':j['fp']-b['fp'],
                                'delta_f1_mean':j['f1_mean']-b['f1_mean']},
            'sha256':{'joint':sha(a.joint),'ranker':sha(a.ranker),'gate':sha(a.gate),
                      'train_telemetry':sha(a.train/'telemetry.csv')},
            'limitations':['Only same-generator synthetic data; no real-world causal validation',
                           'Historical nonuse of seeds cannot be verified',
                           'Joint model trained on signal-selected but evaluated on context-selected candidates',
                           '0.50 is a prespecified default cutoff, not an optimized operating point',
                           'Ground truth used for labels and evaluation only, never inference features',
                           'Candidate rows within a seed are correlated; no independent per-event inference']}
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'per_seed.csv',per_seed)
    write_csv(a.output/'candidate_predictions.csv',candidate_rows)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C50 | FROZEN JOINT CORRECTNESS EVALUATION')
    print(json.dumps({'summary':summaries,'candidate_correctness':report['candidate_correctness'],
                      'paired_aggregate':report['paired_aggregate']},indent=2))
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
