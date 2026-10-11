"""E5-C15 exploratory candidate reranker. Separate train/infer commands.
Train reads truth; infer refuses any truth argument and reads telemetry + trained model.
Requires scikit-learn. Synthetic generator only; not a causal closed-loop test.
"""
import argparse, csv, json, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

SCHEMA={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
FEATURES=['gap','same_source','same_destination','dst_to_src','src_to_dst','same_sensor','same_type','parent_signal','child_signal','candidate_count','gap_rank']

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def candidates(path,threshold,window):
    events=read(path)
    if not events or set(events[0])!=SCHEMA:raise ValueError('Unexpected telemetry schema')
    if len(set(e['event_id'] for e in events))!=len(events):raise ValueError('Duplicate IDs')
    selected=sorted([e for e in events if float(e['signal'])>=threshold],key=lambda e:(int(e['timestamp']),e['event_id']))
    result={}
    for i,child in enumerate(selected):
        ct=int(child['timestamp']); cand=[]
        for parent in reversed(selected[:i]):
            gap=ct-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            cand.append((parent,gap))
        cand.sort(key=lambda z:(z[1],z[0]['event_id']))
        if not cand:continue
        entries=[]
        for rank,(parent,gap) in enumerate(cand,1):
            v=[gap/90.0,
               int(parent['source_host']==child['source_host']),
               int(parent['destination_host']==child['destination_host']),
               int(parent['destination_host']==child['source_host']),
               int(parent['source_host']==child['destination_host']),
               int(parent['sensor']==child['sensor']),
               int(parent['event_type']==child['event_type']),
               float(parent['signal']),float(child['signal']),
               len(cand)/50.0,rank/50.0]
            entries.append((parent['event_id'],gap,v))
        result[child['event_id']]=entries
    return events,selected,result

def train(a):
    truth=read(a.truth)
    gt={r['event_id']:r['predecessor_event_id'] for r in truth if r.get('predecessor_event_id')}
    events,selected,cand=candidates(a.telemetry,a.threshold,a.window)
    X=[];y=[];reachable=0
    for child,entries in cand.items():
        parent=gt.get(child)
        if not parent or parent not in {x[0] for x in entries}:continue
        reachable+=1
        for pid,gap,v in entries:X.append(v);y.append(int(pid==parent))
    if reachable<10:raise RuntimeError('Insufficient reachable training edges')
    model=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=0)
    model.fit(np.asarray(X),np.asarray(y))
    obj={'model':model,'features':FEATURES,'threshold':a.threshold,'window':a.window,'training_reachable':reachable}
    out=Path(a.model);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'wb') as f:pickle.dump(obj,f)
    print('E5-C15 TRAIN | reachable training edges:',reachable,'candidate pairs:',len(y),'positives:',sum(y))
    print('Saved:',out.resolve())

def infer(a):
    with open(a.model,'rb') as f:obj=pickle.load(f) # Only load a locally trained, trusted model file.
    if obj['features']!=FEATURES:raise ValueError('Model feature schema mismatch')
    threshold=obj['threshold'];window=obj['window']
    events,selected,cand=candidates(a.telemetry,threshold,window)
    edges=[]
    for child,entries in cand.items():
        scores=obj['model'].decision_function(np.asarray([v for _,_,v in entries]))
        ranked=sorted(zip(entries,scores),key=lambda z:(-float(z[1]),z[0][1],z[0][0]))
        (pid,gap,_),score=ranked[0]
        edges.append({'parent_event_id':pid,'child_event_id':child,'time_gap':gap})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'predicted_edges.csv',edges,['parent_event_id','child_event_id','time_gap'])
    positive={e['event_id'] for e in selected}
    write(out/'predicted_events.csv',[{'event_id':e['event_id'],'predicted_malicious':int(e['event_id'] in positive)} for e in events],['event_id','predicted_malicious'])
    report={'method':'logistic_pair_reranker_v1','events':len(events),'predicted_positive_events':len(selected),'predicted_edges':len(edges),'threshold':threshold,'window':window,'ground_truth_used':False,'tie_break':'score descending, gap ascending, event_id ascending'}
    (out/'inference_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C15 INFER');print(json.dumps(report,indent=2));print('Saved:',out.resolve())

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train');t.add_argument('--telemetry',required=True);t.add_argument('--truth',required=True);t.add_argument('--model',required=True);t.add_argument('--threshold',type=float,default=.65);t.add_argument('--window',type=int,default=90)
    i=sub.add_parser('infer');i.add_argument('--telemetry',required=True);i.add_argument('--model',required=True);i.add_argument('--output',required=True)
    a=p.parse_args();train(a) if a.cmd=='train' else infer(a)
if __name__=='__main__':main()
