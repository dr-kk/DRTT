import argparse,csv,json
from pathlib import Path

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description='E5-C13 frozen telemetry-only temporal host reconstruction')
    p.add_argument('--telemetry',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threshold',type=float,default=.65);p.add_argument('--window',type=int,default=90)
    a=p.parse_args();events=read(a.telemetry)
    expected={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if set(events[0])!=expected:raise ValueError('Unexpected telemetry schema; hidden columns must not be present')
    if len({e['event_id'] for e in events})!=len(events):raise ValueError('Duplicate event IDs')
    selected=sorted((e for e in events if float(e['signal'])>=a.threshold),key=lambda e:(int(e['timestamp']),e['event_id']))
    preds=[]
    for j,child in enumerate(selected):
        ct=int(child['timestamp']);candidates=[]
        for parent in reversed(selected[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>a.window:break
            if gap<=0:continue
            if (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):
                # Fixed deterministic ranking: closest time, then event ID ascending.
                candidates.append((gap,parent['event_id']))
        if candidates:
            gap,pid=min(candidates,key=lambda x:(x[0],x[1]))
            preds.append({'parent_event_id':pid,'child_event_id':child['event_id'],'time_gap':gap})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'predicted_edges.csv',preds,['parent_event_id','child_event_id','time_gap'])
    write(out/'predicted_events.csv',[{'event_id':e['event_id'],'predicted_malicious':int(e['event_id'] in {x['event_id'] for x in selected})} for e in events],['event_id','predicted_malicious'])
    report={'method':'nearest_host_deterministic_v1','events':len(events),'predicted_positive_events':len(selected),'predicted_edges':len(preds),'threshold':a.threshold,'window':a.window,'tie_break':'min time gap, then parent event ID ascending','ground_truth_used':False}
    (out/'inference_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C13 | TELEMETRY-ONLY INFERENCE');print(json.dumps(report,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
