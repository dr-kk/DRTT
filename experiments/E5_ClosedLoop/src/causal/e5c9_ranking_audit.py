"""E5-C9 diagnostic: telemetry-only ranking, truth used only after rankings frozen.
No parameter fitting; top-k is diagnostic and not a deployable single-edge predictor.
"""
import argparse, csv, json
from pathlib import Path
from collections import deque

def load(p):
    with open(p, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def rank(rows, threshold=.65, window=90):
    allowed={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if not rows or set(rows[0])!=allowed: raise ValueError('Unexpected observable schema')
    stream=sorted(rows,key=lambda e:(int(e['timestamp']),e['event_id']))
    recent=deque(); rankings={}; selected=set()
    for e in stream:
        t=int(e['timestamp'])
        while recent and t-int(recent[0]['timestamp'])>window: recent.popleft()
        if float(e['signal'])<threshold: continue
        selected.add(e['event_id'])
        candidates=[]
        for p in recent:
            gap=t-int(p['timestamp'])
            if gap<=0: continue
            continuity=p['destination_host']==e['source_host']
            same_src=p['source_host']==e['source_host']
            same_dst=p['destination_host']==e['destination_host']
            if not (continuity or same_src or same_dst): continue
            score=(.6 if continuity else 0)+(.25 if same_src or same_dst else 0)+.15*(1-gap/window)
            candidates.append((score,-gap,p['event_id']))
        candidates.sort(reverse=True)
        rankings[e['event_id']]=[p[2] for p in candidates]
        recent.append(e)
    return selected,rankings

def audit(selected, rankings, truth):
    edges=[(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']]
    total=len(edges); detected=0; candidate=0; counts={k:0 for k in (1,3,5,10)}; ranks=[]
    for parent,child in edges:
        if parent not in selected or child not in selected: continue
        detected+=1
        options=rankings.get(child,[])
        if parent not in options: continue
        candidate+=1
        position=options.index(parent)+1
        ranks.append(position)
        for k in counts:
            if position<=k: counts[k]+=1
    return {'true_edges':total,'both_endpoints_detected':detected,'true_parent_in_host_valid_candidates':candidate,
            'candidate_recall':candidate/total if total else 0,
            'mean_true_parent_rank':sum(ranks)/len(ranks) if ranks else None,
            'top_k':{str(k):{'hits':v,'recall_all_edges':v/total if total else 0,
                             'recall_among_reachable':v/candidate if candidate else 0} for k,v in counts.items()}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--telemetry',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selected,rankings=rank(load(a.telemetry))
    result=audit(selected,rankings,load(a.truth))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'ranking_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E5-C9 | FROZEN GRAPH RANKING AUDIT')
    print(json.dumps(result,indent=2))
    print('Saved:',a.output/'ranking_audit.json')
if __name__=='__main__': main()
