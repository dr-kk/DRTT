"""Independent E5-C17 evaluator; method metadata from inference audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e5c13_evaluate import read,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    truth=read(a.truth);base=Path(a.predictions);edges=read(base/'predicted_edges.csv');events=read(base/'predicted_events.csv')
    ids={r['event_id'] for r in truth}
    if len(ids)!=len(truth) or len(events)!=len(truth) or {r['event_id'] for r in events}!=ids:raise ValueError('Event IDs mismatch')
    gt_events={r['event_id'] for r in truth if int(r['malicious'])==1};pred_events={r['event_id'] for r in events if int(r['predicted_malicious'])==1}
    gt_edges={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    pred_edges={(r['parent_event_id'],r['child_event_id']) for r in edges}
    if len(pred_edges)!=len(edges) or any(u not in ids or v not in ids for u,v in pred_edges):raise ValueError('Invalid predicted edges')
    audit=json.loads((base/'inference_audit.json').read_text(encoding='utf-8'))
    report={'event':metrics(len(gt_events&pred_events),len(pred_events),len(gt_events)),'transition':metrics(len(gt_edges&pred_edges),len(pred_edges),len(gt_edges)),'method':audit['method']}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C17 | SEPARATE EVALUATION');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
