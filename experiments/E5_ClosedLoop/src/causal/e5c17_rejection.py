"""E5-C17: Calibrate a no-parent cutoff on development truth; apply without truth.
Use only locally created, trusted pickle models. Never calibrate on final test seeds.
"""
import argparse,csv,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e5c15_reranker import candidates, FEATURES, write, read

def load_model(path):
    with open(path,'rb') as f: obj=pickle.load(f)
    if obj['features']!=FEATURES: raise ValueError('Feature schema mismatch')
    return obj

def score_telemetry(telemetry,obj):
    events,selected,cand=candidates(telemetry,obj['threshold'],obj['window'])
    rows=[]
    for child,entries in cand.items():
        scores=obj['model'].decision_function(np.asarray([v for _,_,v in entries]))
        ranked=sorted(zip(entries,scores),key=lambda z:(-float(z[1]),z[0][1],z[0][0]))
        (pid,gap,_),score=ranked[0]
        rows.append({'parent_event_id':pid,'child_event_id':child,'time_gap':gap,'decision_score':float(score),'candidate_count':len(entries)})
    return events,selected,rows

def evaluate_at(rows,truth_edges,cutoff):
    keep=[r for r in rows if r['decision_score']>=cutoff]
    tp=sum((r['parent_event_id'],r['child_event_id']) in truth_edges for r in keep)
    p=tp/len(keep) if keep else 0.0
    r=tp/len(truth_edges) if truth_edges else 0.0
    return {'cutoff':float(cutoff),'TP':tp,'predicted':len(keep),'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0}

def calibrate(a):
    obj=load_model(a.model)
    _,_,rows=score_telemetry(a.telemetry,obj)
    truth=read(a.truth)
    actual={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    if not actual: raise ValueError('No true edges')
    if len({r['event_id'] for r in truth})!=len(truth):raise ValueError('Duplicate truth IDs')
    vals=sorted({r['decision_score'] for r in rows})
    # Evaluate every achievable operating point (ties are kept together).
    cutoffs=vals+[float('inf')]
    scored=[evaluate_at(rows,actual,c) for c in cutoffs]
    # Conservative tie preference: highest F1, then higher precision, then higher cutoff.
    best=max(scored,key=lambda x:(x['f1'],x['precision'],x['cutoff']))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report={'method':'e5c17_score_rejection_v1','selection':'maximize transition F1 on calibration only','model_path':str(a.model),'calibration_events':len(truth),'calibration_candidate_children':len(rows),'calibration_true_edges':len(actual),'selected_cutoff':best['cutoff'],'calibration_performance':best,'no_rejection_performance':evaluate_at(rows,actual,float('-inf')),'calibration_truth_used':True}
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C17 | CALIBRATION');print(json.dumps(report,indent=2));print('Saved:',out.resolve())

def infer(a):
    obj=load_model(a.model)
    calibration=json.loads(Path(a.calibration).read_text(encoding='utf-8'))
    if calibration['method']!='e5c17_score_rejection_v1':raise ValueError('Wrong calibration file')
    cutoff=float(calibration['selected_cutoff'])
    events,selected,rows=score_telemetry(a.telemetry,obj)
    kept=[r for r in rows if r['decision_score']>=cutoff]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'predicted_edges.csv',kept,['parent_event_id','child_event_id','time_gap','decision_score','candidate_count'])
    positive={e['event_id'] for e in selected}
    write(out/'predicted_events.csv',[{'event_id':e['event_id'],'predicted_malicious':int(e['event_id'] in positive)} for e in events],['event_id','predicted_malicious'])
    audit={'method':'logistic_pair_reranker_with_rejection_v1','events':len(events),'predicted_positive_events':len(selected),'candidate_children':len(rows),'predicted_edges':len(kept),'rejected_edges':len(rows)-len(kept),'decision_cutoff':cutoff,'threshold':obj['threshold'],'window':obj['window'],'ground_truth_used':False,'tie_break':'score descending, gap ascending, event_id ascending'}
    (out/'inference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('E5-C17 | TELEMETRY-ONLY INFERENCE');print(json.dumps(audit,indent=2));print('Saved:',out.resolve())

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--model',required=True);c.add_argument('--telemetry',required=True);c.add_argument('--truth',required=True);c.add_argument('--output',required=True)
    i=sub.add_parser('infer');i.add_argument('--model',required=True);i.add_argument('--telemetry',required=True);i.add_argument('--calibration',required=True);i.add_argument('--output',required=True)
    a=p.parse_args();calibrate(a) if a.command=='calibrate' else infer(a)
if __name__=='__main__':main()
