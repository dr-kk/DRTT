"""E5-C45 frozen end-to-end candidate-validity filtering; synthetic only.
Fixed threshold 0.50, NOT tuned on held-out data. Candidate feature schema exactly
matches E5-C41, applied after context alert selection and before E5-C31 gate.
"""
import argparse,csv,json,pickle,sys,tempfile,hashlib
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate(edges,truth,n_events):
    p=set(edges);tp=len(p&truth);fp=len(p)-tp;rec=tp/len(truth) if truth else 0
    prec=tp/len(p) if p else 0
    return dict(predicted=len(p),tp=tp,fp=fp,precision=prec,recall=rec,
                f1=2*prec*rec/(prec+rec) if prec+rec else 0,
                false_edges_per_1000_events=fp*1000/n_events)

def candidates_with_features(events,selected,ranker,window):
    """Replicate E5-C38 candidate construction and E5-C41 15 features."""
    ordered=sorted((r for r in events if r['event_id'] in selected),key=lambda r:(int(r['timestamp']),r['event_id']))
    records=[]
    for j,child in enumerate(ordered):
        t=int(child['timestamp']);entries=[]
        for parent in reversed(ordered[:j]):
            gap=t-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or
                    parent['source_host']==child['source_host'] or
                    parent['destination_host']==child['destination_host']):continue
            entries.append((parent,gap))
        entries.sort(key=lambda v:(v[1],v[0]['event_id']))
        if not entries:continue
        vectors=[]
        for ix,(parent,gap) in enumerate(entries):
            vectors.append([gap/90.,float(parent['source_host']==child['source_host']),
              float(parent['destination_host']==child['destination_host']),
              float(parent['destination_host']==child['source_host']),
              float(parent['source_host']==child['destination_host']),
              float(parent['sensor']==child['sensor']),float(parent['event_type']==child['event_type']),
              float(parent['signal']),float(child['signal']),len(entries)/50.,ix/50.])
        v=np.asarray(vectors,dtype=float)
        scores=np.asarray(ranker['model'].decision_function(v),dtype=float).reshape(-1)
        order=sorted(range(len(entries)),key=lambda i:(-float(scores[i]),entries[i][1],entries[i][0]['event_id']))
        best=order[0];second=order[1] if len(order)>1 else best
        vec=v[best];gaps=np.asarray([e[1] for e in entries],dtype=float)
        valid_features=[np.log1p(len(entries)),gaps.min()/window,gaps.mean()/window,
          v[:,7].min(),v[:,7].max(),vec[8],scores[best],scores[best]-scores[second],scores.std(),*vec[1:7]]
        gate_features=[scores[best],scores[best]-scores[second],scores.mean(),scores.std(),
          np.log1p(len(entries)),entries[best][1]/window,vec[7],vec[8],*vec[1:7]]
        if len(valid_features)!=15 or len(gate_features)!=14:raise AssertionError('Feature count')
        records.append({'child':child['event_id'],'parent':entries[best][0]['event_id'],
                        'valid_features':valid_features,'gate_features':gate_features})
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--start-seed',type=int,default=4500)
    ap.add_argument('--num-seeds',type=int,default=5)
    args=ap.parse_args()
    if args.num_seeds<2 or args.start_seed<4500:ap.error('Use at least 2 seeds starting >=4500')
    if args.output.exists() and any(args.output.iterdir()):ap.error('Output directory must be empty')
    sys.path.insert(0,str(args.scripts.resolve()))
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with args.models.open('rb') as f:bundle=pickle.load(f)  # trusted local artifacts
    with args.ranker.open('rb') as f:ranker=pickle.load(f)
    with args.gate.open('rb') as f:gate=pickle.load(f)
    if bundle['ranker_sha256']!=sha(args.ranker) or gate['e5c15_sha256']!=sha(args.ranker):raise ValueError('Ranker hash mismatch')
    if bundle['features']!=c41.FEATURES or gate['features']!=c38.GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature schema mismatch')
    if int(gate['window'])!=90 or int(ranker['window'])!=90 or float(ranker['threshold'])!=.65:raise ValueError('Unexpected frozen settings')
    if set(bundle['models'])!={'balanced','unweighted','balanced_sigmoid_cv5'}:raise ValueError('Unexpected models')
    # Refit only the original E5-C38 context detector on E5-C15 training telemetry.
    # This detector is not calibrated or tuned on any E5-C45 labels.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    train=c38.read(args.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(args.train/'ground_truth.csv')}
    features=c38.extract(train,90)
    ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([features[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];manifest=[]
    for seed in range(args.start_seed,args.start_seed+args.num_seeds):
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        x=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([x[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
        records=candidates_with_features(events,chosen,ranker,90)
        # Reproduction check: no-filter baseline must match the original E5-C38 routine.
        expected,audit=c38.top_predictions(events,chosen,ranker,gate,base,90)
        if len(records)!=audit['candidate_children']:raise AssertionError('Candidate count differs from E5-C38')
        if records:
            gate_probs=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records]))[:,1]
            baseline={(r['parent'],r['child']) for r,p in zip(records,gate_probs) if p>=gate['cutoff']}
        else:
            gate_probs=np.array([]);baseline=set()
        if baseline!=set(expected):raise AssertionError(f'Seed {seed}: baseline mismatch with E5-C38')
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        variants={'context_frozen_gate':baseline}
        if records:
            V=np.asarray([r['valid_features'] for r in records])
            for name,model in bundle['models'].items():
                vp=model.predict_proba(V)[:,1]
                variants[f'context_{name}_validity_0_5']={
                    (r['parent'],r['child']) for r,g,v in zip(records,gate_probs,vp)
                    if g>=gate['cutoff'] and v>=.5}
        else:
            for name in bundle['models']:variants[f'context_{name}_validity_0_5']=set()
        for name,edges in variants.items():
            metrics=evaluate(edges,truth,len(events))
            rows.append({'seed':seed,'pipeline':name,'alerts':n,'candidate_children':len(records),**metrics})
        manifest.append({'seed':seed,'events':len(events),'truth_edges':len(truth),'alerts':n})
        print(f'Seed {seed}: '+', '.join(f'{name} TP/FP={evaluate(e,truth,len(events))["tp"]}/{evaluate(e,truth,len(events))["fp"]}' for name,e in variants.items()))
    write_csv(args.output/'per_seed.csv',rows)
    summary={}
    for name in sorted({r['pipeline'] for r in rows}):
        rr=[r for r in rows if r['pipeline']==name]
        tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=sum(r['recall'] for r in rr)/len(rr)
        summary[name]={'tp':tp,'fp':fp,'precision_pooled':precision,'recall_mean':recall,
           'f1_mean':float(np.mean([r['f1'] for r in rr])),
           'false_edges_per_1000_events_mean':float(np.mean([r['false_edges_per_1000_events'] for r in rr]))}
    report={'experiment':'E5-C45','scope':'frozen candidate-validity models, fixed 0.50 cutoff, fresh specified synthetic seeds',
        'seed_manifest':manifest,'summary':summary,'sha256':{'models':sha(args.models),'ranker':sha(args.ranker),'gate':sha(args.gate)},
        'limitations':['Synthetic generator only; no operational evidence','Historical nonuse of seeds unverified',
          'Validity model trained on signal-selected candidates but applied to context-selected candidates (distribution shift)',
          '0.50 is a fixed probability threshold, not optimized for causal-link F1',
          'Context detector refitted deterministically on original E5-C15 data',
          'No claim of superior recall or overall F1 before evaluation']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C45 | FROZEN END-TO-END SYNTHETIC COMPARISON')
    print(json.dumps(summary,indent=2))
    print('Saved:',args.output.resolve())

if __name__=='__main__':main()
