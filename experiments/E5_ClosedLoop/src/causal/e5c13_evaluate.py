import argparse,csv,json
from pathlib import Path

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def metrics(tp,npred,ntrue):
    precision=tp/npred if npred else 0.;recall=tp/ntrue if ntrue else 0.
    return {'TP':tp,'predicted':npred,'ground_truth':ntrue,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}

def main():
    p=argparse.ArgumentParser(description='E5-C13 separate offline evaluator')
    p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();truth=read(a.truth);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    edges=read(Path(a.predictions)/'predicted_edges.csv');events=read(Path(a.predictions)/'predicted_events.csv')
    ids={r['event_id'] for r in truth}
    if len(ids)!=len(truth):raise ValueError('Duplicate truth event IDs')
    if {r['event_id'] for r in events}!=ids or len(events)!=len(truth):raise ValueError('Prediction and truth event IDs mismatch')
    actual_events={r['event_id'] for r in truth if int(r['malicious'])==1}
    predicted_events={r['event_id'] for r in events if int(r['predicted_malicious'])==1}
    actual_edges={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    predicted_edges={(r['parent_event_id'],r['child_event_id']) for r in edges}
    if len(predicted_edges)!=len(edges):raise ValueError('Duplicate predicted edges')
    if any(u not in ids or v not in ids for u,v in predicted_edges):raise ValueError('Unknown event in predicted edges')
    report={'event':metrics(len(actual_events & predicted_events),len(predicted_events),len(actual_events)),'transition':metrics(len(actual_edges & predicted_edges),len(predicted_edges),len(actual_edges)),'method':'nearest_host_deterministic_v1'}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C13 | SEPARATE GROUND-TRUTH EVALUATION');print(json.dumps(report,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
