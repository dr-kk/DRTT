"""E5-C31 development-only no-parent gate. Does not read sealed test data.
Uses locally trusted E5-C15 pickle; train on e5c15_train, calibrate on
 e5c17_calibration. Offline truth never enters inference feature generation.
"""
import argparse, csv, json, pickle, hashlib, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES=['best_score','score_margin','score_mean','score_std','log_candidate_count',
          'best_gap_norm','best_parent_signal','child_signal','same_source',
          'same_destination','dst_to_src','src_to_dst','same_sensor','same_type']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_truth(path):
    with open(path,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not rows or not {'event_id','predecessor_event_id'} <= set(rows[0]):
        raise ValueError('Invalid truth schema')
    if len({r['event_id'] for r in rows})!=len(rows):raise ValueError('Duplicate truth IDs')
    return {(r['predecessor_event_id'],r['event_id']) for r in rows if r['predecessor_event_id']}, {r['event_id']:r['predecessor_event_id'] for r in rows}

def score_dataset(dataset,model,base):
    # No ground truth is read here: safe for later sealed inference.
    events,selected,cand=base.candidates(dataset/'telemetry.csv',model['threshold'],model['window'])
    output=[]
    for child,entries in cand.items():
        scores=np.asarray(model['model'].decision_function(np.asarray([v for _,_,v in entries])),dtype=float).reshape(-1)
        ranked=sorted(zip(entries,scores),key=lambda z:(-float(z[1]),z[0][1],z[0][0]))
        (parent,gap,vec),top=ranked[0]
        second=float(ranked[1][1]) if len(ranked)>1 else float(top)
        x=[float(top),float(top)-second,float(np.mean(scores)),float(np.std(scores)),
           float(np.log1p(len(entries))),float(gap)/float(model['window']),
           float(vec[7]),float(vec[8]),*map(float,vec[1:7])]
        if len(x)!=len(FEATURES):raise RuntimeError('Feature mismatch')
        output.append({'parent':parent,'child':child,'score':float(top),'x':x})
    return output,{'events':len(events),'selected_events':len(selected),'candidate_children':len(output)}

def metrics(rows,truth,keep):
    chosen=[r for r,k in zip(rows,keep) if k]
    tp=sum((r['parent'],r['child']) in truth for r in chosen)
    n=len(chosen);p=tp/n if n else 0.;r=tp/len(truth) if truth else 0.
    return {'predicted':n,'tp':tp,'fp':n-tp,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-dataset',type=Path,required=True)
    p.add_argument('--calibration-dataset',type=Path,required=True)
    p.add_argument('--scripts',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--baseline-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-predictions',type=int,default=30)
    a=p.parse_args()
    if a.min_predictions<1:p.error('min-predictions must be positive')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    with a.model.open('rb') as f:model=pickle.load(f)  # trusted local model only
    if model['features']!=base.FEATURES:raise ValueError('E5-C15 feature mismatch')
    old=json.loads(a.baseline_calibration.read_text(encoding='utf-8'))
    if old.get('method')!='e5c17_score_rejection_v1':raise ValueError('Invalid baseline calibration')
    cutoff=float(old['selected_cutoff'])
    train,train_audit=score_dataset(a.train_dataset,model,base)
    cal,cal_audit=score_dataset(a.calibration_dataset,model,base)
    train_truth,_=read_truth(a.train_dataset/'ground_truth.csv')
    cal_truth,_=read_truth(a.calibration_dataset/'ground_truth.csv')
    y=np.asarray([int((r['parent'],r['child']) in train_truth) for r in train],dtype=int)
    if len(np.unique(y))!=2:raise RuntimeError('Need both positive and negative training labels')
    clf=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=0)
    clf.fit(np.asarray([r['x'] for r in train]),y)
    probs=clf.predict_proba(np.asarray([r['x'] for r in cal]))[:,1]
    # Optimize exact-edge F1 on development calibration only, with min support.
    # Tie break: precision then higher threshold (more conservative).
    points=[]
    for threshold in sorted(set(map(float,probs))):
        keep=probs>=threshold
        if int(keep.sum())<a.min_predictions:continue
        m=metrics(cal,cal_truth,keep)
        points.append({'cutoff':threshold,**m})
    if not points:raise RuntimeError('No eligible calibration operating points')
    best=max(points,key=lambda z:(z['f1'],z['precision'],z['cutoff']))
    old_m=metrics(cal,cal_truth,[r['score']>=cutoff for r in cal])
    new_m=metrics(cal,cal_truth,probs>=best['cutoff'])
    # Save model and report only after calibration. No sealed test set is touched.
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'no_parent_gate.pkl').open('wb') as f:
        pickle.dump({'model':clf,'features':FEATURES,'cutoff':best['cutoff'],
                     'e5c15_sha256':sha256(a.model),'threshold':model['threshold'],
                     'window':model['window']},f)
    with (a.output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['cutoff','predicted','tp','fp','precision','recall','f1'])
        writer.writeheader();writer.writerows(points)
    report={'design':'Development-only gate for correctness of top-ranked causal parent; not independently validated',
            'train':train_audit,'train_positive_top1':int(y.sum()),
            'calibration':cal_audit,'calibration_true_edges':len(cal_truth),
            'baseline_cutoff':cutoff,'baseline':old_m,
            'new_gate_cutoff':best['cutoff'],'new_gate':new_m,
            'training_label_definition':'top-ranked parent equals true predecessor (not general existence of a parent)',
            'features':FEATURES,'min_predictions':a.min_predictions,
            'training_calibration_sha256':{'train_telemetry':sha256(a.train_dataset/'telemetry.csv'),
            'cal_telemetry':sha256(a.calibration_dataset/'telemetry.csv')},
            'limitations':['Calibration F1 is in-sample and cannot establish test improvement',
            'Frozen pairwise ranker may select wrong parent even when a true parent exists',
            'Synthetic datasets only','No fresh test labels read','Historical seed disjointness not proven']}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C31 | DEVELOPMENT-ONLY NO-PARENT GATE')
    print(json.dumps({'train':train_audit,'calibration':cal_audit,'baseline':old_m,'new_gate':new_m,
                      'cutoff':best['cutoff']},indent=2))
    print('Saved:',a.output.resolve())
    print('Do not change model based on sealed test results.')

if __name__=='__main__':main()
