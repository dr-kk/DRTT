"""E5-C10 diagnostic: quantify observable separation of true and false predecessor candidates.
Ground truth is used for retrospective audit only; this is not an inference model.
"""
import argparse, csv, json, statistics
from pathlib import Path
from collections import deque

def load(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def describe(rows):
    if not rows: return {'n':0}
    gaps=[x['gap'] for x in rows]
    return {'n':len(rows),'gap_mean':round(statistics.mean(gaps),3),
            'gap_median':statistics.median(gaps),
            'continuity_rate':round(sum(x['continuity'] for x in rows)/len(rows),4),
            'same_source_rate':round(sum(x['same_source'] for x in rows)/len(rows),4),
            'same_destination_rate':round(sum(x['same_destination'] for x in rows)/len(rows),4),
            'same_sensor_rate':round(sum(x['same_sensor'] for x in rows)/len(rows),4),
            'same_event_type_rate':round(sum(x['same_event_type'] for x in rows)/len(rows),4)}

def run(telemetry, truth, threshold, window):
    expected={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if not telemetry or set(telemetry[0]) != expected: raise ValueError('Unexpected telemetry columns')
    ordered=sorted(telemetry,key=lambda e:(int(e['timestamp']),e['event_id']))
    parents={r['event_id']:r['predecessor_event_id'] for r in truth if r['predecessor_event_id']}
    recent=deque(); positives=[]; negatives=[]; child_count=0; reachable=0; comparisons=[]
    for child in ordered:
        t=int(child['timestamp'])
        while recent and t-int(recent[0]['timestamp'])>window: recent.popleft()
        if float(child['signal'])<threshold: continue
        if child['event_id'] in parents: child_count+=1
        candidates=[]
        for p in recent:
            gap=t-int(p['timestamp'])
            if gap<=0: continue
            cont=int(p['destination_host']==child['source_host'])
            same_src=int(p['source_host']==child['source_host'])
            same_dst=int(p['destination_host']==child['destination_host'])
            if not (cont or same_src or same_dst): continue
            feat={'gap':gap,'continuity':cont,'same_source':same_src,
                  'same_destination':same_dst,
                  'same_sensor':int(p['sensor']==child['sensor']),
                  'same_event_type':int(p['event_type']==child['event_type'])}
            candidates.append((p['event_id'],feat))
        parent=parents.get(child['event_id'])
        if parent and any(eid==parent for eid,_ in candidates):
            reachable+=1
            true_feat=next(f for eid,f in candidates if eid==parent)
            positives.append(true_feat)
            false=[f for eid,f in candidates if eid!=parent]
            negatives.extend(false)
            comparisons.append({'false_count':len(false),
                'true_gap_smaller_fraction':sum(true_feat['gap']<f['gap'] for f in false)/len(false) if false else None})
        recent.append(child)
    result={'detected_children_with_truth_parent':child_count,'reachable_true_edges':reachable,
            'true_edge_features':describe(positives),'competing_false_edge_features':describe(negatives),
            'mean_competitors_per_reachable_child':round(statistics.mean(x['false_count'] for x in comparisons),3) if comparisons else None,
            'mean_fraction_false_with_larger_gap':round(statistics.mean(x['true_gap_smaller_fraction'] for x in comparisons if x['true_gap_smaller_fraction'] is not None),4) if comparisons else None,
            'warning':'Ground-truth-assisted exploratory audit, not a blind performance evaluation'}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--telemetry',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.65)
    p.add_argument('--window',type=int,default=90)
    a=p.parse_args()
    result=run(load(a.telemetry),load(a.truth),a.threshold,a.window)
    a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/'evidence_audit.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E5-C10 | OBSERVABLE EVIDENCE AUDIT')
    print(json.dumps(result,indent=2))
    print('Saved:',dest)
if __name__=='__main__':main()
