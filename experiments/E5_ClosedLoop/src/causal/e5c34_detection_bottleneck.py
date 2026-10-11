"""E5-C34: development-only detector threshold and true-edge reachability audit.
No model fitting, no threshold selection, no use of sealed test data.
"""
import argparse
import csv
import json
from pathlib import Path

OBSERVED = {'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
TRUTH_REQUIRED = {'event_id','malicious','predecessor_event_id'}

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def safe_div(a,b):
    return a/b if b else 0.0

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True,help='Development dataset directory only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thresholds',type=float,nargs='+',default=[.30,.40,.50,.60,.65,.70,.80])
    p.add_argument('--window',type=int,default=90)
    a=p.parse_args()
    if a.window<1 or any(not 0<=t<=1 for t in a.thresholds):
        p.error('window must be positive; thresholds must be between 0 and 1')
    if 'e5c31_fresh_test' in str(a.dataset).lower() or any(f'e5c18_{s}' in str(a.dataset).lower() for s in range(2040,2045)):
        p.error('Refusing known held-out/test datasets: use e5c15_train or e5c17_calibration')
    if a.dataset.resolve()==a.output.resolve():
        p.error('Output cannot be the dataset directory')
    telemetry=read_csv(a.dataset/'telemetry.csv')
    truth=read_csv(a.dataset/'ground_truth.csv')
    if not telemetry or set(telemetry[0])!=OBSERVED:
        raise ValueError('Unexpected telemetry schema')
    if not truth or not TRUTH_REQUIRED.issubset(truth[0]):
        raise ValueError('Unexpected ground-truth schema')
    events={r['event_id']:r for r in telemetry}
    gt={r['event_id']:r for r in truth}
    if len(events)!=len(telemetry) or len(gt)!=len(truth) or set(events)!=set(gt):
        raise ValueError('Duplicate or mismatched event IDs')
    for r in telemetry:
        float(r['signal']);int(r['timestamp'])
    malicious={eid for eid,r in gt.items() if int(r['malicious'])==1}
    true_edges={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    if not all(parent in events and child in events for parent,child in true_edges):
        raise ValueError('Truth contains nonexistent event IDs')
    rows=[]
    for threshold in sorted(set(a.thresholds)):
        selected={eid for eid,r in events.items() if float(r['signal'])>=threshold}
        tp=len(selected & malicious)
        fp=len(selected-malicious)
        fn=len(malicious-selected)
        tn=len(events)-tp-fp-fn
        both={(parent,child) for parent,child in true_edges if parent in selected and child in selected}
        in_window={(parent,child) for parent,child in both if 0<int(events[child]['timestamp'])-int(events[parent]['timestamp'])<=a.window}
        host_ok={(parent,child) for parent,child in in_window if (
            events[parent]['destination_host']==events[child]['source_host'] or
            events[parent]['source_host']==events[child]['source_host'] or
            events[parent]['destination_host']==events[child]['destination_host'])}
        precision=safe_div(tp,tp+fp)
        recall=safe_div(tp,tp+fn)
        rows.append({
            'threshold':threshold,'events':len(events),'malicious_events':len(malicious),
            'selected_events':len(selected),'event_tp':tp,'event_fp':fp,'event_fn':fn,'event_tn':tn,
            'event_precision':precision,'event_recall':recall,
            'event_f1':safe_div(2*precision*recall,precision+recall),
            'event_false_positive_rate':safe_div(fp,fp+tn),
            'false_alerts_per_1000_events':1000*safe_div(fp,len(events)),
            'truth_edges':len(true_edges),'both_endpoints_detected':len(both),
            'both_endpoints_recall_ceiling':safe_div(len(both),len(true_edges)),
            'time_window_eligible_edges':len(in_window),
            'host_candidate_eligible_edges':len(host_ok),
            'candidate_recall_ceiling':safe_div(len(host_ok),len(true_edges)),
            'detection_lost_edges':len(true_edges)-len(both),
            'window_lost_edges':len(both)-len(in_window),
            'host_lost_edges':len(in_window)-len(host_ok)
        })
    a.output.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]);write_csv(a.output/'threshold_sweep.csv',rows,fields)
    report={
        'experiment':'E5-C34 development-only signal threshold audit',
        'dataset':str(a.dataset),'events':len(events),'malicious_events':len(malicious),
        'truth_edges':len(true_edges),'window':a.window,
        'selection_policy':'No threshold selected; report complete prespecified sweep',
        'results':rows,
        'limitations':['Development labels used only for offline diagnostics',
            'Synthetic signal threshold is not a trained detector',
            'Endpoint detection and candidate reachability are optimistic ceilings, not causal edge prediction',
            'Do not choose a new cutoff using previously inspected E5-C18 or E5-C32 test seeds']}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C34 | DEVELOPMENT-ONLY DETECTION BOTTLENECK')
    print('threshold | event P | event R | false alerts/1k | reachable true edges')
    for r in rows:
        print(f"{r['threshold']:8.2f} | {r['event_precision']:7.3f} | {r['event_recall']:7.3f} | {r['false_alerts_per_1000_events']:15.2f} | {r['host_candidate_eligible_edges']:4d}/{len(true_edges)}")
    print('Saved:',a.output.resolve())

if __name__=='__main__':
    main()
