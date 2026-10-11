"""E5-C38: frozen forensic reconstruction under matched alert budgets.
Requires trusted local E5-C15 and E5-C31 pickle artifacts. Synthetic only.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
from collections import defaultdict,deque
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE_FEATURES=['best_score','score_margin','score_mean','score_std','log_candidate_count','best_gap_norm','best_parent_signal','child_signal','same_source','same_destination','dst_to_src','src_to_dst','same_sensor','same_type']

def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def extract(events,window):
    ordered=sorted(events,key=lambda r:(int(r['timestamp']),r['event_id']))
    history=deque();src=defaultdict(int);dst=defaultdict(int);pair=defaultdict(int);out={};i=0
    while i<len(ordered):
        t=int(ordered[i]['timestamp'])
        while history and history[0][0]<t-window:
            _,s,d=history.popleft();src[s]-=1;dst[d]-=1;pair[(s,d)]-=1
        j=i
        while j<len(ordered) and int(ordered[j]['timestamp'])==t:j+=1
        for r in ordered[i:j]:
            s,d=r['source_host'],r['destination_host']
            out[r['event_id']]=[float(r['signal']),src[s],dst[d],pair[(s,d)]]
        for r in ordered[i:j]:
            s,d=r['source_host'],r['destination_host'];history.append((t,s,d));src[s]+=1;dst[d]+=1;pair[(s,d)]+=1
        i=j
    return out

def top_predictions(events,chosen,ranker,gate,base,window):
    # Replicate E5-C15 candidates, replacing only selection; never replace raw signal.
    selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
    out=[]; candidate_count=0
    for j,child in enumerate(selected):
        ct=int(child['timestamp']);entries=[]
        for parent in reversed(selected[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            entries.append((parent,gap))
        entries.sort(key=lambda x:(x[1],x[0]['event_id']))
        if not entries:continue
        candidate_count+=1
        vectors=[]
        for ix,(p,gap) in enumerate(entries):
            vec=[gap/90.,float(p['source_host']==child['source_host']),float(p['destination_host']==child['destination_host']),float(p['destination_host']==child['source_host']),float(p['source_host']==child['destination_host']),float(p['sensor']==child['sensor']),float(p['event_type']==child['event_type']),float(p['signal']),float(child['signal']),len(entries)/50.,ix/50.]
            vectors.append(vec)
        scores=np.asarray(ranker['model'].decision_function(np.asarray(vectors)),dtype=float).reshape(-1)
        ranked=sorted(zip(entries,vectors,scores),key=lambda z:(-float(z[2]),z[0][1],z[0][0]['event_id']))
        ((p,gap),vec,top)=ranked[0]
        second=float(ranked[1][2]) if len(ranked)>1 else float(top)
        x=[float(top),float(top)-second,float(np.mean(scores)),float(np.std(scores)),float(np.log1p(len(entries))),float(gap)/float(window),float(vec[7]),float(vec[8]),*map(float,vec[1:7])]
        if len(x)!=len(GATE_FEATURES):raise RuntimeError('Gate feature dimension mismatch')
        out.append((p['event_id'],child['event_id'],x))
    if not out:return [],{'selected_events':len(selected),'candidate_children':candidate_count,'gate_probability_mean':None}
    probs=gate['model'].predict_proba(np.asarray([r[2] for r in out]))[:,1]
    kept=[(p,c) for (p,c,_),prob in zip(out,probs) if prob>=gate['cutoff']]
    return kept,{'selected_events':len(selected),'candidate_children':candidate_count,'gate_probability_mean':float(np.mean(probs))}

def metric(pred,truth,events):
    tp=len(set(pred)&truth);n=len(pred);p=tp/n if n else 0.;r=tp/len(truth) if truth else 0.
    return {'predicted':n,'tp':tp,'fp':n-tp,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'false_edges_per_1000_events':(n-tp)*1000/len(events)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--output',type=Path,default=Path('results/causal/e5c38'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--window',type=int,default=90)
    args=ap.parse_args()
    sys.path.insert(0,str(args.scripts.resolve()))
    import e5c5_overlapping_generator as gen
    import e5c15_reranker as base
    with args.ranker.open('rb') as f:ranker=pickle.load(f)
    with args.gate.open('rb') as f:gate=pickle.load(f)
    if gate['e5c15_sha256']!=digest(args.ranker):raise ValueError('Frozen E5-C15 SHA256 mismatch')
    if gate['features']!=GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature schema mismatch')
    if int(gate['window'])!=args.window or int(ranker['window'])!=args.window:raise ValueError('Window mismatch')
    if float(gate['threshold'])!=.65 or float(ranker['threshold'])!=.65:raise ValueError('Unexpected original threshold')
    tr=read(args.train/'telemetry.csv');ty={r['event_id']:int(r['malicious']) for r in read(args.train/'ground_truth.csv')}
    ca=read(args.calibration/'telemetry.csv');cy={r['event_id']:int(r['malicious']) for r in read(args.calibration/'ground_truth.csv')}
    tx=extract(tr,args.window);cx=extract(ca,args.window)
    ids=[r['event_id'] for r in tr];X=np.asarray([tx[i] for i in ids]);y=np.asarray([ty[i] for i in ids])
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    model.fit(X,y)
    cids=[r['event_id'] for r in ca];cprobs=model.predict_proba(np.asarray([cx[i] for i in cids]))[:,1]
    # Freeze calibration alert fraction based solely on original signal threshold .65.
    budget=sum(float(r['signal'])>=.65 for r in ca)/len(ca)
    # Exact alert count per test dataset, not test-label-dependent; rank probabilities deterministically.
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    results=[];rows=[]
    for seed in range(3800,3805):
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events];x=extract(events,args.window)
        probs=model.predict_proba(np.asarray([x[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        # Matching actual baseline alert count on each test realization is an oracle-free budget rule.
        # It is not a pre-fixed probability cutoff; disclose that distinction.
        ranked=sorted(zip(ids,probs),key=lambda v:(-float(v[1]),v[0]))
        selected_context={i for i,_ in ranked[:n]}
        selected_signal={r['event_id'] for r in events if float(r['signal'])>=.65}
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        for name,chosen in [('signal',selected_signal),('context',selected_context)]:
            pred,audit=top_predictions(events,chosen,ranker,gate,base,args.window)
            m=metric(pred,truth,events)
            rows.append({'seed':seed,'pipeline':name,**m,**audit})
        print(f'Seed {seed}: signal F1={rows[-2]["f1"]:.4f}, context F1={rows[-1]["f1"]:.4f}; alerts={n}')
    import statistics as st
    summary={}
    for name in ('signal','context'):
        rr=[r for r in rows if r['pipeline']==name]
        summary[name]={k:float(st.mean(r[k] for r in rr)) for k in ('precision','recall','f1','false_edges_per_1000_events','selected_events','candidate_children')}
        summary[name]['predicted']=sum(r['predicted'] for r in rr);summary[name]['tp']=sum(r['tp'] for r in rr);summary[name]['fp']=sum(r['fp'] for r in rr)
    delta={k:summary['context'][k]-summary['signal'][k] for k in ('precision','recall','f1','false_edges_per_1000_events')}
    with (output/'per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'experiment':'E5-C38','scope':'fresh synthetic evaluation, no test-label tuning','seeds':list(range(3800,3805)),'generator':{'campaigns':100,'duration':2400,'background':10000},'alert_budget':'context selects top-N per test realization, N equals number of signal>=0.65 alerts (unlabeled budget matching)','calibration_signal_alert_fraction':budget,'gate_cutoff':gate['cutoff'],'model_sha256':{'ranker':digest(args.ranker),'gate':digest(args.gate)},'train_calibration_hashes':{'train_telemetry':digest(args.train/'telemetry.csv'),'calibration_telemetry':digest(args.calibration/'telemetry.csv')},'mean_by_pipeline':summary,'paired_mean_delta':delta,'limitations':['Same synthetic generator as development','Historical seed disjointness not independently verified','Gate calibration may shift under contextual selection','Per-test alert count is matched using unlabeled signal scores; not fixed probability threshold','Model comparison is not an operational deployment test']}
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C38 | FROZEN END-TO-END SYNTHETIC TEST')
    print(json.dumps({'mean_by_pipeline':summary,'paired_mean_delta':delta},indent=2))
    print('Saved:',output.resolve())
if __name__=='__main__':main()
