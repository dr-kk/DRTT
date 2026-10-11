import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ALLOWED = {'event_id','campaign_id','timestamp','source_host','destination_host','sensor','event_type','signal'}

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def infer(events, threshold=0.65, max_gap=90):
    if not events:
        raise ValueError('Empty telemetry')
    fields=set(events[0])
    if fields != ALLOWED:
        raise ValueError(f'Unexpected telemetry schema: {sorted(fields ^ ALLOWED)}')
    grouped=defaultdict(list)
    for e in events:
        grouped[e['campaign_id']].append(e)
    predictions=[]
    edges=[]
    for campaign, rows in sorted(grouped.items()):
        rows=sorted(rows, key=lambda e:(int(e['timestamp']), e['event_id']))
        selected=[]
        for e in rows:
            confidence=float(e['signal'])
            positive=confidence >= threshold
            predictions.append({'event_id':e['event_id'],'campaign_id':campaign,'confidence':round(confidence,6),'predicted_malicious':int(positive)})
            if not positive:
                continue
            candidates=[]
            for prev in selected:
                gap=int(e['timestamp'])-int(prev['timestamp'])
                if gap < 0 or gap > max_gap:
                    continue
                # Observable entity continuity, not hidden predecessor labels.
                continuity=(prev['destination_host']==e['source_host'])
                same_host=(prev['source_host']==e['source_host'] or prev['destination_host']==e['destination_host'])
                if not (continuity or same_host):
                    continue
                score=(0.6 if continuity else 0.3)+(0.25 if same_host else 0)+0.15*(1-gap/max_gap)
                candidates.append((score,-gap,prev['event_id']))
            if candidates:
                score, _, predecessor=max(candidates)
                edges.append({'campaign_id':campaign,'predecessor_event_id':predecessor,'successor_event_id':e['event_id'],'edge_score':round(score,6)})
            selected.append(e)
    return predictions, edges

def evaluate(predictions, edges, truth):
    gt={r['event_id']:int(r['malicious']) for r in truth}
    if set(gt)!={r['event_id'] for r in predictions}:
        raise ValueError('Mismatched evaluation IDs')
    tp=fp=tn=fn=0
    for p in predictions:
        y=gt[p['event_id']]; yh=int(p['predicted_malicious'])
        if y and yh: tp+=1
        elif not y and yh: fp+=1
        elif not y: tn+=1
        else: fn+=1
    actual={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    predicted={(e['predecessor_event_id'],e['successor_event_id']) for e in edges}
    edge_tp=len(actual & predicted)
    ep=edge_tp/len(predicted) if predicted else 0.0
    er=edge_tp/len(actual) if actual else 0.0
    return {'event':{'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'fpr':fp/(fp+tn) if fp+tn else 0},'transition':{'true_edges':len(actual),'predicted_edges':len(predicted),'TP':edge_tp,'precision':ep,'recall':er,'f1':2*ep*er/(ep+er) if ep+er else 0}}

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--telemetry',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--truth',type=Path,default=None,help='Evaluation only; never used by infer()')
    ap.add_argument('--threshold',type=float,default=0.65)
    ap.add_argument('--max-gap',type=int,default=90)
    args=ap.parse_args()
    if not 0<=args.threshold<=1 or args.max_gap<=0: ap.error('Invalid threshold or max-gap')
    args.output.mkdir(parents=True,exist_ok=True)
    predictions,edges=infer(read_csv(args.telemetry),args.threshold,args.max_gap)
    write_csv(args.output/'event_predictions.csv',predictions,['event_id','campaign_id','confidence','predicted_malicious'])
    write_csv(args.output/'predicted_edges.csv',edges,['campaign_id','predecessor_event_id','successor_event_id','edge_score'])
    print('E5-C2 | GRAPH-CONSTRAINED RECONSTRUCTION')
    print('Predicted events:',len(predictions),'Predicted edges:',len(edges))
    print('Inference completed using telemetry only.')
    if args.truth:
        report=evaluate(predictions,edges,read_csv(args.truth))
        with open(args.output/'metrics.json','w',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))
    print('Saved:',args.output)

if __name__=='__main__':main()
