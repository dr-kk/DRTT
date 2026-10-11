"""E5-C11: fixed, telemetry-only predecessor rankings; optional truth for audit only.
Exploratory diagnostic, not a held-out confirmatory test. No parameter fitting.
"""
import argparse, csv, json
from pathlib import Path
from collections import deque

def load(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def infer(rows, threshold=0.65, window=90):
    allowed={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if not rows or set(rows[0]) != allowed: raise ValueError('Unexpected telemetry columns')
    stream=sorted(rows,key=lambda e:(int(e['timestamp']),e['event_id']))
    recent=deque(); ranks={'frozen_graph':{},'nearest_host':{},'temporal_host':{}}
    for e in stream:
        t=int(e['timestamp'])
        while recent and t-int(recent[0]['timestamp'])>window: recent.popleft()
        if float(e['signal'])<threshold: continue
        candidates=[]
        for p in recent:
            gap=t-int(p['timestamp'])
            if gap<=0: continue
            continuity=p['destination_host']==e['source_host']
            same_src=p['source_host']==e['source_host']
            same_dst=p['destination_host']==e['destination_host']
            if not (continuity or same_src or same_dst): continue
            frozen=(.6 if continuity else 0)+(.25 if same_src or same_dst else 0)+.15*(1-gap/window)
            # Fixed, interpretable baseline: time first, structural evidence for ties.
            structural=2*int(continuity)+int(same_src)+int(same_dst)
            candidates.append((p['event_id'],gap,frozen,structural))
        cid=e['event_id']
        ranks['frozen_graph'][cid]=[v[0] for v in sorted(candidates,key=lambda v:(-v[2],v[1],v[0]))]
        ranks['nearest_host'][cid]=[v[0] for v in sorted(candidates,key=lambda v:(v[1],v[0]))]
        ranks['temporal_host'][cid]=[v[0] for v in sorted(candidates,key=lambda v:(v[1],-v[3],v[0]))]
        recent.append(e)
    return ranks

def evaluate(ranks,truth):
    edges=[(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']]
    output={'true_edges':len(edges),'methods':{}}
    for method,rankings in ranks.items():
        predicted={(opts[0],child) for child,opts in rankings.items() if opts}
        gold=set(edges); tp=len(predicted & gold)
        p=tp/len(predicted) if predicted else 0
        r=tp/len(gold) if gold else 0
        reachable=sum(parent in rankings.get(child,[]) for parent,child in edges)
        top={str(k):sum(parent in rankings.get(child,[])[:k] for parent,child in edges) for k in (1,3,5,10)}
        output['methods'][method]={'predicted_edges':len(predicted),'TP':tp,'precision':p,'recall':r,
          'f1':2*p*r/(p+r) if p+r else 0,'reachable_true_edges':reachable,'top_k_hits':top}
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--telemetry',type=Path,required=True)
    parser.add_argument('--truth',type=Path,help='Optional audit only; never passed into infer()')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    rankings=infer(load(a.telemetry))
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rank in rankings.items():
        write(a.output/f'{name}_edges.csv',
          [{'predecessor_event_id':options[0],'event_id':child} for child,options in rank.items() if options],
          ['predecessor_event_id','event_id'])
    print('E5-C11 | FROZEN TELEMETRY-ONLY RANKINGS')
    print('Predictions written before ground-truth audit')
    if a.truth:
        report=evaluate(rankings,load(a.truth))
        (a.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    print('Saved:',a.output.resolve())
if __name__=='__main__': main()
