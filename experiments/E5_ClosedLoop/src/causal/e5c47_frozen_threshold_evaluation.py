"""E5-C47: frozen E5-C46 operating-point evaluation on fresh synthetic seeds.
No threshold fitting or tuning; uses E5-C45's exact candidate construction.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,default=Path('results/causal/e5c46/selection.json'))
    p.add_argument('--models',type=Path,default=Path('results/causal/e5c43/models.pkl'))
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c47'))
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory must be empty; previous results are preserved')
    sel=json.loads(a.selection.read_text(encoding='utf-8'))
    if sel.get('experiment')!='E5-C46' or sel.get('selected_model')!='unweighted':raise ValueError('Unexpected selection')
    cutoff=float(sel['selected_threshold'])
    if not (0<cutoff<1):raise ValueError('Invalid threshold')
    for key,path in [('models',a.models),('ranker',a.ranker),('gate',a.gate),('train_telemetry',a.train/'telemetry.csv')]:
        if sel['sha256'][key]!=sha(path):raise ValueError(f'{key} SHA256 differs from frozen selection')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.models.open('rb') as f:bundle=pickle.load(f)
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if bundle['ranker_sha256']!=sha(a.ranker) or gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Ranker hash mismatch')
    if bundle['features']!=c41.FEATURES or gate['features']!=c38.GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature mismatch')
    if float(ranker['threshold'])!=.65 or int(ranker['window'])!=90 or int(gate['window'])!=90:raise ValueError('Unexpected ranker/gate config')
    model=bundle['models']['unweighted']
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    feat=c38.extract(train,90)
    ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    rows=[];manifest=[]
    for seed in range(4700,4705):
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        x=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([x[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
        records=c45.candidates_with_features(events,chosen,ranker,90)
        expected,audit=c38.top_predictions(events,chosen,ranker,gate,base,90)
        if len(records)!=audit['candidate_children']:raise AssertionError('Candidate count mismatch')
        if records:
            g=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records]))[:,1]
            v=model.predict_proba(np.asarray([r['valid_features'] for r in records]))[:,1]
            baseline={(r['parent'],r['child']) for r,prob in zip(records,g) if prob>=gate['cutoff']}
            filtered={(r['parent'],r['child']) for r,gp,vp in zip(records,g,v) if gp>=gate['cutoff'] and vp>=cutoff}
        else:baseline=set();filtered=set()
        if baseline!=set(expected):raise AssertionError(f'Seed {seed}: baseline mismatch with E5-C38')
        if not filtered.issubset(baseline):raise AssertionError('Filtered not subset')
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        b=c45.evaluate(baseline,truth,len(events));f=c45.evaluate(filtered,truth,len(events))
        rows.extend([{'seed':seed,'pipeline':'context_frozen_gate','alerts':n,'candidate_children':len(records),**b},
                     {'seed':seed,'pipeline':'context_unweighted_frozen_e5c46','alerts':n,'candidate_children':len(records),**f}])
        manifest.append({'seed':seed,'events':len(events),'truth_edges':len(truth),'alerts':n})
        print(f'Seed {seed}: baseline TP/FP={b["tp"]}/{b["fp"]}; frozen-filter TP/FP={f["tp"]}/{f["fp"]}; retention={f["tp"]/b["tp"]:.2%} FP reduction={(b["fp"]-f["fp"])/b["fp"]:.2%}')
    summary={}
    for name in ('context_frozen_gate','context_unweighted_frozen_e5c46'):
        rr=[r for r in rows if r['pipeline']==name]
        tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr)
        summary[name]={'tp':tp,'fp':fp,'precision_pooled':tp/(tp+fp) if tp+fp else 0,
                       'recall_mean':float(np.mean([r['recall'] for r in rr])),
                       'f1_mean':float(np.mean([r['f1'] for r in rr])),
                       'false_edges_per_1000_events_mean':float(np.mean([r['false_edges_per_1000_events'] for r in rr]))}
    b=summary['context_frozen_gate'];f=summary['context_unweighted_frozen_e5c46']
    paired=[]
    for seed in range(4700,4705):
        bb=next(r for r in rows if r['seed']==seed and r['pipeline']=='context_frozen_gate')
        ff=next(r for r in rows if r['seed']==seed and r['pipeline']=='context_unweighted_frozen_e5c46')
        paired.append({'seed':seed,'tp_retention':ff['tp']/bb['tp'] if bb['tp'] else None,
                       'fp_reduction':(bb['fp']-ff['fp'])/bb['fp'] if bb['fp'] else None,
                       'delta_precision':ff['precision']-bb['precision'],
                       'delta_recall':ff['recall']-bb['recall'],'delta_f1':ff['f1']-bb['f1'],
                       'delta_false_edges_per_1000_events':ff['false_edges_per_1000_events']-bb['false_edges_per_1000_events']})
    report={'experiment':'E5-C47','scope':'frozen threshold on five new same-generator synthetic seeds',
            'threshold':cutoff,'selection_sha256':sha(a.selection),'manifest':manifest,'summary':summary,
            'paired_aggregate':{'tp_retention':f['tp']/b['tp'] if b['tp'] else None,
                                'fp_reduction':(b['fp']-f['fp'])/b['fp'] if b['fp'] else None,
                                'delta_f1_mean':f['f1_mean']-b['f1_mean']},
            'sha256':{'models':sha(a.models),'ranker':sha(a.ranker),'gate':sha(a.gate)},
            'limitations':['Only synthetic data from the same generator','Historical seed independence unverified',
                           'No real operational or causal attribution validation',
                           'Validity model trained on signal-selected, applied to context-selected candidates',
                           'Do not tune the selected threshold after inspecting this evaluation']}
    a.output.mkdir(parents=True,exist_ok=True)
    csv_write(a.output/'per_seed.csv',rows);csv_write(a.output/'paired_differences.csv',paired)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C47 | FROZEN THRESHOLD EVALUATION')
    print(json.dumps({'threshold':cutoff,'summary':summary,'paired_aggregate':report['paired_aggregate']},indent=2))
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
