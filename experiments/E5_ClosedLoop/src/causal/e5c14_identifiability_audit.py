"""E5-C14: ground-truth-assisted identifiability audit, NOT an inference model."""
import argparse,csv,json,statistics
from collections import Counter
from pathlib import Path

def load(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--telemetry',required=True)
    ap.add_argument('--truth',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--threshold',type=float,default=.65)
    ap.add_argument('--window',type=int,default=90)
    a=ap.parse_args()
    events=load(a.telemetry);truth=load(a.truth)
    expected={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if not events or set(events[0])!=expected:raise ValueError('Telemetry schema invalid or contains hidden fields')
    byid={e['event_id']:e for e in events}
    if len(byid)!=len(events):raise ValueError('Duplicate event IDs')
    positives=sorted((e for e in events if float(e['signal'])>=a.threshold),key=lambda e:(int(e['timestamp']),e['event_id']))
    selected={e['event_id'] for e in positives}
    parents={r['event_id']:r['predecessor_event_id'] for r in truth if r['predecessor_event_id']}
    if set(parents)-set(byid) or set(parents.values())-set(byid):raise ValueError('Truth references unknown IDs')
    results=Counter();candidate_counts=[];rank_counts=[];tie_counts=[];examples=[]
    for j,child in enumerate(positives):
        cid=child['event_id'];pid=parents.get(cid)
        if not pid:continue
        results['true_edges_with_detected_child']+=1
        if pid not in selected:continue
        results['both_endpoints_detected']+=1
        ct=int(child['timestamp'])
        candidates=[]
        for parent in reversed(positives[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>a.window:break
            if gap<=0:continue
            if (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):
                candidates.append(parent)
        if pid not in {p['event_id'] for p in candidates}:continue
        results['reachable_edges']+=1
        candidate_counts.append(len(candidates))
        def features(p):
            return (ct-int(p['timestamp']),
                    int(p['destination_host']==child['source_host']),
                    int(p['source_host']==child['source_host']),
                    int(p['destination_host']==child['destination_host']),
                    int(p['sensor']==child['sensor']),
                    int(p['event_type']==child['event_type']))
        target=byid[pid];ftrue=features(target)
        nearest=sorted(candidates,key=lambda p:(ct-int(p['timestamp']),p['event_id']))
        rank=next(k for k,p in enumerate(nearest,1) if p['event_id']==pid)
        rank_counts.append(rank)
        if rank==1:results['nearest_top1_correct']+=1
        if rank<=3:results['nearest_top3_contains']+=1
        if rank<=5:results['nearest_top5_contains']+=1
        if rank<=10:results['nearest_top10_contains']+=1
        same_gap=sum(features(p)[0]==ftrue[0] for p in candidates)
        same_full=sum(features(p)==ftrue for p in candidates)
        if same_gap>1:results['true_parent_has_equal_gap_competitor']+=1
        if same_full>1:results['true_parent_has_exact_feature_twin']+=1
        tie_counts.append(same_full)
        if len(examples)<5 and (same_full>1 or rank>5):
            examples.append({'child_event_id':cid,'true_parent_event_id':pid,'nearest_rank':rank,'candidate_count':len(candidates),'equal_gap_candidates':same_gap,'identical_feature_candidates':same_full})
    total=len(parents);reachable=results['reachable_edges']
    report={'dataset':str(a.telemetry),'true_edges':total,'threshold':a.threshold,'window':a.window,
            'counts':dict(results),'candidate_recall_all_edges':reachable/total if total else 0,
            'mean_host_valid_candidates':statistics.mean(candidate_counts) if candidate_counts else 0,
            'median_nearest_true_parent_rank':statistics.median(rank_counts) if rank_counts else None,
            'nearest_top1_recall_among_reachable':results['nearest_top1_correct']/reachable if reachable else 0,
            'exact_feature_twin_fraction_among_reachable':results['true_parent_has_exact_feature_twin']/reachable if reachable else 0,
            'examples':examples,
            'limitations':'Uses ground truth for analysis; identical chosen features do not establish full statistical non-identifiability; generator is synthetic.'}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'identifiability_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C14 | IDENTIFIABILITY AUDIT');print(json.dumps(report,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
