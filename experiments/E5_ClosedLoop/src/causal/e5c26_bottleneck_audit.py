#!/usr/bin/env python3
"""E5-C26: offline edge-recovery loss decomposition and threshold sensitivity.

Ground truth is used exclusively for evaluation. This is an optimistic
endpoint/time-window diagnostic, not an oracle for all candidate constraints.
"""
import argparse,csv,json,statistics,hashlib
from pathlib import Path

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def save(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--e5c25',required=True,help='Existing E5-C25 output folder containing per_dataset.csv')
    p.add_argument('--output',required=True)
    p.add_argument('--thresholds',nargs='+',type=float,default=[0.3,0.4,0.5,0.6,0.65,0.7,0.8])
    p.add_argument('--window',type=float,default=90)
    a=p.parse_args()
    if not all(0<=t<=1 for t in a.thresholds) or a.window<=0:p.error('Thresholds must be [0,1] and window positive')
    e5=Path(a.e5c25)/'per_dataset.csv'
    if not e5.is_file():p.error('Missing E5-C25 per_dataset.csv: '+str(e5))
    previous=read(e5);lookup={(r['dataset'],r['method']):r for r in previous}
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    rows=[];digest_rows=[]
    for foldername in a.datasets:
        folder=Path(foldername); telepath=folder/'telemetry.csv';gtpath=folder/'ground_truth.csv'
        if not telepath.is_file() or not gtpath.is_file():p.error('Missing telemetry/ground_truth in '+str(folder))
        tele=read(telepath);truth=read(gtpath)
        if not tele or not truth:raise ValueError('Empty dataset: '+str(folder))
        byid={r['event_id']:r for r in tele}
        if len(byid)!=len(tele):raise ValueError('Duplicate telemetry IDs: '+str(folder))
        if set(byid)!={r['event_id'] for r in truth}:raise ValueError('Truth/telemetry ID mismatch: '+str(folder))
        edges=[(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']]
        if any(u not in byid or v not in byid for u,v in edges):raise ValueError('Missing edge endpoints')
        digest_rows.append(dict(dataset=folder.name,telemetry_sha256=hashlib.sha256(telepath.read_bytes()).hexdigest(),truth_sha256=hashlib.sha256(gtpath.read_bytes()).hexdigest()))
        for threshold in a.thresholds:
            detected={eid for eid,r in byid.items() if float(r['signal'])>=threshold}
            both=[(u,v) for u,v in edges if u in detected and v in detected]
            eligible=[(u,v) for u,v in both if 0<float(byid[v]['timestamp'])-float(byid[u]['timestamp'])<=a.window]
            total=len(edges);both_n=len(both);elig_n=len(eligible)
            row=dict(dataset=folder.name,threshold=threshold,window=a.window,total_events=len(tele),detected_events=len(detected),true_edges=total,
                detection_lost_edges=total-both_n,detected_endpoint_edges=both_n,
                time_window_lost_edges=both_n-elig_n,eligible_endpoint_time_edges=elig_n,
                optimistic_recall_ceiling=elig_n/total if total else 0)
            if abs(threshold-.65)<1e-9 and abs(a.window-90)<1e-9:
                for method in ('temporal_host','learned_rejection'):
                    key=(folder.name,method)
                    if key not in lookup:raise ValueError('Missing E5-C25 result for '+str(key))
                    r=lookup[key];tp=int(r['TP'])
                    if tp>elig_n:
                        raise ValueError(f'{key}: TP {tp} > endpoint/time eligible {elig_n}; audit assumptions mismatch')
                    row[method+'_tp']=tp
                    row[method+'_eligible_missed']=elig_n-tp
                    row[method+'_eligible_recovery_fraction']=tp/elig_n if elig_n else 0
            rows.append(row)
    save(output/'threshold_sensitivity.csv',rows)
    save(output/'input_hashes.csv',digest_rows)
    at065=[r for r in rows if abs(r['threshold']-.65)<1e-9]
    summary={'datasets':len(a.datasets),'thresholds':a.thresholds,'window':a.window,
      'at_065':{'true_edges':sum(r['true_edges'] for r in at065),'detection_lost_edges':sum(r['detection_lost_edges'] for r in at065),
        'time_window_lost_edges':sum(r['time_window_lost_edges'] for r in at065),
        'eligible_endpoint_time_edges':sum(r['eligible_endpoint_time_edges'] for r in at065),
        'optimistic_recall_ceiling':sum(r['eligible_endpoint_time_edges'] for r in at065)/sum(r['true_edges'] for r in at065) if at065 else None},
      'limitations':['Offline evaluator accesses private ground truth; inference must not',
        'Endpoint/time eligibility is an optimistic proxy, not complete candidate-generator reachability',
        'Eligible missed edges combine candidate schema, ranking, and rejection losses; cannot distinguish them without candidate logs',
        'Threshold sweep is diagnostic only; do not select a new threshold on these test seeds',
        'All datasets are synthetic E5-C5 campaigns']}
    if at065:
        for method in ('temporal_host','learned_rejection'):
            summary['at_065'][method+'_tp']=sum(r[method+'_tp'] for r in at065)
            summary['at_065'][method+'_eligible_missed']=sum(r[method+'_eligible_missed'] for r in at065)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C26 | EDGE RECOVERY BOTTLENECK AUDIT');print(json.dumps(summary,indent=2));print('Saved:',output.resolve())
if __name__=='__main__':main()

