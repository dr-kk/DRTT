"""E5-C39: frozen gate distribution-shift audit, same synthetic seeds as E5-C38.
No training, threshold tuning, or model changes. Ground truth used only for audit.
Run alongside e5c38_end_to_end.py in src/causal.
"""
import argparse,csv,json,pickle,sys,statistics
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import e5c38_end_to_end as c38

def diagnose(events,chosen,ranker,gate,truth,window):
    selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
    selected_ids={r['event_id'] for r in selected}
    true_parent={child:parent for parent,child in truth}
    records=[]
    for j,child in enumerate(selected):
        ct=int(child['timestamp']);entries=[]
        for parent in reversed(selected[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            entries.append((parent,gap))
        entries.sort(key=lambda z:(z[1],z[0]['event_id']))
        if not entries:continue
        vectors=[]
        for ix,(p,gap) in enumerate(entries):
            vectors.append([gap/90.,float(p['source_host']==child['source_host']),float(p['destination_host']==child['destination_host']),float(p['destination_host']==child['source_host']),float(p['source_host']==child['destination_host']),float(p['sensor']==child['sensor']),float(p['event_type']==child['event_type']),float(p['signal']),float(child['signal']),len(entries)/50.,ix/50.])
        scores=np.asarray(ranker['model'].decision_function(np.asarray(vectors)),dtype=float).reshape(-1)
        ranked=sorted(zip(entries,vectors,scores),key=lambda z:(-float(z[2]),z[0][1],z[0][0]['event_id']))
        ((p,gap),vec,top)=ranked[0]
        second=float(ranked[1][2]) if len(ranked)>1 else float(top)
        x=[float(top),float(top)-second,float(np.mean(scores)),float(np.std(scores)),float(np.log1p(len(entries))),float(gap)/float(window),float(vec[7]),float(vec[8]),*map(float,vec[1:7])]
        if len(x)!=len(c38.GATE_FEATURES):raise ValueError('Gate feature length mismatch')
        parent_id=p['event_id'];child_id=child['event_id'];real_parent=true_parent.get(child_id)
        available=real_parent is not None and real_parent in selected_ids
        in_candidates=real_parent is not None and any(q['event_id']==real_parent for q,_ in entries)
        records.append({'child':child_id,'parent':parent_id,'true_parent':real_parent or '',
                        'true_parent_selected':int(available),'true_parent_candidate':int(in_candidates),
                        'top1_correct':int((parent_id,child_id) in truth),'candidate_count':len(entries),
                        'gate_features':x})
    if records:
        probabilities=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records]))[:,1]
        for r,pr in zip(records,probabilities):
            r['gate_probability']=float(pr);r['accepted']=int(pr>=gate['cutoff'])
    return records

def summarize(rows):
    accepted=[r for r in rows if r['accepted']]
    correct=[r for r in accepted if r['top1_correct']]
    wrong=[r for r in accepted if not r['top1_correct']]
    return {'candidate_children':len(rows),'accepted':len(accepted),'accepted_tp':len(correct),'accepted_fp':len(wrong),
            'top1_correct_before_gate':sum(r['top1_correct'] for r in rows),
            'true_parent_in_candidates':sum(r['true_parent_candidate'] for r in rows),
            'accepted_wrong_no_true_parent_candidate':sum(not r['true_parent_candidate'] for r in wrong),
            'accepted_wrong_with_true_parent_candidate':sum(r['true_parent_candidate'] for r in wrong),
            'gate_probability_mean':statistics.mean(r['gate_probability'] for r in rows) if rows else None,
            'gate_probability_accepted_mean':statistics.mean(r['gate_probability'] for r in accepted) if accepted else None,
            'candidate_count_mean':statistics.mean(r['candidate_count'] for r in rows) if rows else None,
            'score_margin_mean':statistics.mean(r['gate_features'][1] for r in rows) if rows else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--output',type=Path,default=Path('results/causal/e5c39'))
    ap.add_argument('--reference',type=Path,default=Path('results/causal/e5c38/per_seed.csv'))
    ap.add_argument('--window',type=int,default=90)
    args=ap.parse_args()
    with args.ranker.open('rb') as f:ranker=pickle.load(f)
    with args.gate.open('rb') as f:gate=pickle.load(f)
    if c38.digest(args.ranker)!=gate['e5c15_sha256']:raise ValueError('Ranker hash mismatch')
    if gate['features']!=c38.GATE_FEATURES:raise ValueError('Gate schema mismatch')
    if int(gate['window'])!=args.window or int(ranker['window'])!=args.window:raise ValueError('Window mismatch')
    if float(gate['threshold'])!=.65 or float(ranker['threshold'])!=.65:raise ValueError('Threshold mismatch')
    import e5c5_overlapping_generator as gen
    train=c38.read(args.train/'telemetry.csv');truth_train={r['event_id']:int(r['malicious']) for r in c38.read(args.train/'ground_truth.csv')}
    features=c38.extract(train,args.window)
    ids=[r['event_id'] for r in train]
    X=np.asarray([features[i] for i in ids]);y=np.asarray([truth_train[i] for i in ids])
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    model.fit(X,y)
    reference={}
    if args.reference.exists():
        for r in c38.read(args.reference):reference[(int(r['seed']),r['pipeline'])]=r
    args.output.mkdir(parents=True,exist_ok=True)
    details=[];reports=[]
    for seed in range(3800,3805):
        events,gt=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events];feat=c38.extract(events,args.window)
        probs=model.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        context={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
        signal={r['event_id'] for r in events if float(r['signal'])>=.65}
        truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r['predecessor_event_id']}
        for name,chosen in [('signal',signal),('context',context)]:
            rows=diagnose(events,chosen,ranker,gate,truth,args.window)
            summary=summarize(rows)
            ref=reference.get((seed,name))
            if ref:
                for key,col in [('accepted','predicted'),('accepted_tp','tp'),('accepted_fp','fp'),('candidate_children','candidate_children')]:
                    if summary[key]!=int(float(ref[col])):raise RuntimeError(f'E5-C38 reproduction mismatch seed={seed} {name} {key}: {summary[key]} != {ref[col]}')
            reports.append({'seed':seed,'pipeline':name,**summary})
            for r in rows:
                details.append({'seed':seed,'pipeline':name,**{k:v for k,v in r.items() if k!='gate_features'},
                                **dict(zip(c38.GATE_FEATURES,r['gate_features']))})
        print(f'Seed {seed}: signal TP/FP={reports[-2]["accepted_tp"]}/{reports[-2]["accepted_fp"]}; context TP/FP={reports[-1]["accepted_tp"]}/{reports[-1]["accepted_fp"]}')
    with (args.output/'candidate_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    with (args.output/'per_seed_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(reports[0]));w.writeheader();w.writerows(reports)
    totals={}
    for name in ('signal','context'):
        rr=[r for r in reports if r['pipeline']==name]
        totals[name]={k:sum(r[k] for r in rr) for k in ('candidate_children','accepted','accepted_tp','accepted_fp','top1_correct_before_gate','true_parent_in_candidates','accepted_wrong_no_true_parent_candidate','accepted_wrong_with_true_parent_candidate')}
    result={'experiment':'E5-C39','scope':'post-hoc audit of already evaluated E5-C38 synthetic seeds; not new held-out validation','ranker_sha256':c38.digest(args.ranker),'gate_sha256':c38.digest(args.gate),'reference_checked':bool(reference),'totals':totals,'per_seed':reports,'note':'Ground truth used solely for post-hoc diagnostics; no model changes or threshold selection.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E5-C39 | FROZEN GATE DISTRIBUTION-SHIFT AUDIT')
    print(json.dumps(totals,indent=2))
    print('Saved:',args.output.resolve())
if __name__=='__main__':main()
