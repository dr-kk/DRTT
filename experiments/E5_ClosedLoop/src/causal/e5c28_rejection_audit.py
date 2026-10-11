"""E5-C28: frozen E5-C17 cutoff audit on existing held-out synthetic datasets.
No training, no recalibration, no ground truth during inference.
Only load locally trusted pickle model. Python >=3.9, numpy, sklearn.
"""
import argparse,csv,json,math,sys,hashlib
from pathlib import Path
from collections import defaultdict

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)
def safe_float(v):
    if isinstance(v,str) and v.lower() in ('inf','infinity'):return float('inf')
    return float(v)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--scripts',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--calibration',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(a.scripts).resolve()))
    from e5c17_rejection import load_model,score_telemetry
    obj=load_model(a.model)
    cal=json.loads(Path(a.calibration).read_text(encoding='utf-8'))
    if cal.get('method')!='e5c17_score_rejection_v1':raise ValueError('Unexpected calibration method')
    cutoff=safe_float(cal['selected_cutoff'])
    if math.isnan(cutoff):raise ValueError('NaN cutoff')
    out=Path(a.output)
    if out.resolve() in [Path(x).resolve() for x in a.datasets]:raise ValueError('Output cannot be dataset path')
    out.mkdir(parents=True,exist_ok=True)
    details=[];aggregate=defaultdict(int);per_seed=[]
    for d in a.datasets:
        folder=Path(d)
        telemetry=folder/'telemetry.csv';truth=folder/'ground_truth.csv'
        if not telemetry.is_file() or not truth.is_file():raise FileNotFoundError(f'Missing telemetry/truth: {folder}')
        events,selected,scored=score_telemetry(str(telemetry),obj) # telemetry only
        # Private evaluation begins here. Truth never reaches model or scoring.
        gt={(r['predecessor_event_id'],r['event_id']) for r in read_csv(truth) if r.get('predecessor_event_id')}
        if not gt:raise ValueError('No truth edges in '+str(folder))
        selected_ids={r['event_id'] for r in selected}
        pre=len(scored);kept=[r for r in scored if float(r['decision_score'])>=cutoff]
        tp_before=sum((r['parent_event_id'],r['child_event_id']) in gt for r in scored)
        tp_after=sum((r['parent_event_id'],r['child_event_id']) in gt for r in kept)
        n=len(events);ntrue=len(gt)
        def metrics(tp,pred):
            precision=tp/pred if pred else 0.0;recall=tp/ntrue if ntrue else 0.0
            return dict(predicted=pred,true_positive=tp,false_positive=pred-tp,precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,false_edges_per_1000_events=1000*(pred-tp)/n if n else 0.0)
        b=metrics(tp_before,pre);r=metrics(tp_after,len(kept))
        row={'dataset':folder.name,'events':n,'selected_events':len(selected),'true_edges':ntrue,'candidate_children':pre,'cutoff':cutoff,
             'telemetry_sha256':hashlib.sha256(telemetry.read_bytes()).hexdigest(),
             **{'before_'+k:v for k,v in b.items()},**{'after_'+k:v for k,v in r.items()},
             'rejected_total':pre-len(kept),'rejected_true':tp_before-tp_after,
             'rejected_false':(pre-tp_before)-(len(kept)-tp_after)}
        per_seed.append(row)
        for key in ('events','selected_events','true_edges','candidate_children','before_predicted','before_true_positive','before_false_positive','after_predicted','after_true_positive','after_false_positive','rejected_total','rejected_true','rejected_false'):
            aggregate[key]+=row[key]
        print(f"{folder.name}: before P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}; after P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")
    write_csv(out/'per_dataset.csv',per_seed)
    ntrue=aggregate['true_edges'];n=aggregate['events']
    def combined(prefix):
        tp=aggregate[prefix+'_true_positive'];pred=aggregate[prefix+'_predicted'];fp=aggregate[prefix+'_false_positive']
        precision=tp/pred if pred else 0;recall=tp/ntrue if ntrue else 0
        return {'predicted':pred,'true_positive':tp,'false_positive':fp,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'false_edges_per_1000_events':1000*fp/n if n else 0}
    report={'design':'Frozen E5-C15 model and E5-C17 development-calibrated cutoff; ground truth evaluation only','datasets':len(per_seed),'total_events':n,'total_truth_edges':ntrue,'model_threshold':obj['threshold'],'model_window':obj['window'],'frozen_cutoff':cutoff,
            'before_rejection':combined('before'),'after_rejection':combined('after'),
            'rejected_total':aggregate['rejected_total'],'rejected_true':aggregate['rejected_true'],'rejected_false':aggregate['rejected_false'],
            'fraction_false_among_rejected':aggregate['rejected_false']/aggregate['rejected_total'] if aggregate['rejected_total'] else None,
            'calibration_method':cal['method'],
            'limitations':['Five previously used E5-C18 synthetic test datasets; not fresh external validation','The original cutoff was chosen to maximize development F1, not precision','Offline ground truth is used exclusively after telemetry-only inference','Do not optimize cutoff on these test seeds','Check original training/calibration seed disjointness separately']}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C28 | FROZEN REJECTION AUDIT')
    print(json.dumps(report,indent=2))
    print('Saved:',out.resolve())
if __name__=='__main__':main()
