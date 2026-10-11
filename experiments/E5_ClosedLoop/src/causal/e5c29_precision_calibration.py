"""E5-C29: development-only precision-constrained cutoff feasibility audit.
Frozen E5-C15 model. No test data. Does not modify prior calibration.
Only load a locally created trusted pickle model.
"""
import argparse, csv, hashlib, json, math, sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    if not rows: return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def wilson_lower(tp, n, z=1.959963984540054):
    if n==0:return 0.0
    p=tp/n
    return (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration-dataset',required=True)
    p.add_argument('--train-dataset',required=True)
    p.add_argument('--scripts',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=float,nargs='+',default=[0.2,0.3,0.5])
    p.add_argument('--min-predictions',type=int,default=30)
    a=p.parse_args()
    if a.min_predictions<1:raise ValueError('min-predictions must be positive')
    if any(not 0<t<=1 for t in a.targets):raise ValueError('Targets must be in (0,1]')
    cal=Path(a.calibration_dataset);train=Path(a.train_dataset);out=Path(a.output)
    if cal.resolve()==train.resolve():raise ValueError('Train and calibration paths are identical')
    if out.resolve() in (cal.resolve(),train.resolve()):raise ValueError('Output must be separate')
    for d in (cal,train):
        for filename in ('telemetry.csv','ground_truth.csv'):
            if not (d/filename).is_file():raise FileNotFoundError(d/filename)
    hashes={name:{fn:sha(d/fn) for fn in ('telemetry.csv','ground_truth.csv')} for name,d in [('train',train),('calibration',cal)]}
    for fn in ('telemetry.csv','ground_truth.csv'):
        if hashes['train'][fn]==hashes['calibration'][fn]:
            raise ValueError('Identical train and calibration '+fn)
    sys.path.insert(0,str(Path(a.scripts).resolve()))
    from e5c17_rejection import load_model,score_telemetry
    obj=load_model(a.model)
    events,selected,scored=score_telemetry(str(cal/'telemetry.csv'),obj)
    # Ground truth is loaded ONLY after inference scores are frozen.
    truth_rows=read(cal/'ground_truth.csv')
    if len({r['event_id'] for r in truth_rows})!=len(truth_rows):raise ValueError('Duplicate truth event IDs')
    gt={(r['predecessor_event_id'],r['event_id']) for r in truth_rows if r.get('predecessor_event_id')}
    if not gt:raise ValueError('No true edges')
    labeled=[]
    for r in scored:
        s=float(r['decision_score'])
        if not math.isfinite(s):raise ValueError('Non-finite score')
        labeled.append((s,int((r['parent_event_id'],r['child_event_id']) in gt)))
    labeled.sort(key=lambda x:-x[0])
    total=len(labeled); positives=sum(y for _,y in labeled)
    points=[];tp=0;i=0
    # Evaluate each attainable nonempty operating point; score ties are retained together.
    while i<total:
        score=labeled[i][0]
        while i<total and labeled[i][0]==score:
            tp+=labeled[i][1];i+=1
        precision=tp/i;recall=tp/len(gt)
        points.append(dict(cutoff=score,predicted=i,true_positive=tp,false_positive=i-tp,
                           precision=precision,recall=recall,
                           f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                           precision_wilson95_lower=wilson_lower(tp,i),
                           retained_fraction=i/total))
    targets=[]
    for target in a.targets:
        feasible=[x for x in points if x['predicted']>=a.min_predictions and x['precision']>=target]
        # Highest recall, then precision, then stricter cutoff; calibration-only exploratory selection.
        best=max(feasible,key=lambda x:(x['recall'],x['precision'],x['cutoff'])) if feasible else None
        targets.append(dict(target_precision=target,min_predictions=a.min_predictions,
                            calibration_feasible=bool(best),selected=best,
                            **({'note':'No cutoff meets target and minimum prediction count on calibration data'} if not best else {})))
    out.mkdir(parents=True,exist_ok=True)
    write(out/'calibration_operating_points.csv',points)
    report={'design':'Calibration-only exploratory precision-constrained cutoff feasibility; not test validation',
            'dataset':cal.name,'training_dataset':train.name,'events':len(events),'selected_events':len(selected),
            'truth_edges':len(gt),'scored_children':total,'correct_top1_before_rejection':positives,
            'model_threshold':obj['threshold'],'model_window':obj['window'],
            'targets':targets,'train_calibration_file_hashes':hashes,
            'provenance':'Different file hashes verified; original generator seed disjointness NOT established',
            'limitations':['Same calibration dataset was previously used to select E5-C17 F1 cutoff',
                           'These are in-sample exploratory operating points, not independent validation',
                           'Minimum support does not ensure a precision guarantee',
                           'Do not tune or evaluate on previously inspected E5-C18 seeds',
                           'Synthetic correlated events are not independently verified real-world causal chains']}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('E5-C29 | CALIBRATION-ONLY PRECISION FEASIBILITY')
    for t in targets:
        b=t['selected']
        print(f"target {t['target_precision']:.0%}: "+(f"FEASIBLE ON CALIBRATION | cutoff={b['cutoff']:.6g} P={b['precision']:.3%} R={b['recall']:.3%} predicted={b['predicted']} Wilson95lower={b['precision_wilson95_lower']:.3%}" if b else 'INFEASIBLE'))
    print('Saved:',out.resolve())

if __name__=='__main__':main()
