"""E5-C58: train-only gate for frozen E5-C56; calibrate on E5-C17 only.
No new-seed evaluation, no test-seed access, no overwriting existing results.
Uses synthetic telemetry and trusted local pickle artifacts.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE_FEATURES=['best_score','score_margin','score_mean','score_std','log_candidate_count','best_gap_norm','best_parent_signal','child_signal','same_source','same_destination','dst_to_src','src_to_dst','same_sensor','same_type']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
    if fields is None:fields=list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def top_rows(events,chosen,ranker,window):
    """Replicate E5-C38/E5-C56 context candidates, with rank index starting at zero."""
    selected=sorted((e for e in events if e['event_id'] in chosen),key=lambda e:(int(e['timestamp']),e['event_id']))
    rows=[]
    for j,child in enumerate(selected):
        t=int(child['timestamp']);pairs=[]
        for parent in reversed(selected[:j]):
            gap=t-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            pairs.append((parent,gap))
        pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
        if not pairs:continue
        vec=[]
        for i,(parent,gap) in enumerate(pairs):
            vec.append([gap/90.,float(parent['source_host']==child['source_host']),float(parent['destination_host']==child['destination_host']),float(parent['destination_host']==child['source_host']),float(parent['source_host']==child['destination_host']),float(parent['sensor']==child['sensor']),float(parent['event_type']==child['event_type']),float(parent['signal']),float(child['signal']),len(pairs)/50.,i/50.])
        x=np.asarray(vec,dtype=float)
        if x.shape[1]!=11:raise ValueError('Candidate feature mismatch')
        z=x if len(ranker['features'])==11 else x[:,:-1]
        scores=np.asarray(ranker['model'].decision_function(z),dtype=float).reshape(-1)
        order=sorted(range(len(pairs)),key=lambda i:(-float(scores[i]),pairs[i][1],pairs[i][0]['event_id']))
        best=order[0];parent,gap=pairs[best];top=float(scores[best]);second=float(scores[order[1]]) if len(order)>1 else top
        v=x[best]
        features=[top,top-second,float(np.mean(scores)),float(np.std(scores)),float(np.log1p(len(pairs))),float(gap)/float(window),float(v[7]),float(v[8]),*map(float,v[1:7])]
        if len(features)!=len(GATE_FEATURES):raise ValueError('Gate feature mismatch')
        rows.append({'parent':parent['event_id'],'child':child['event_id'],'features':features})
    return rows

def truth_edges(path):
    truth=read(path)
    return {(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}

def metrics(rows,truth,keep):
    selected=[r for r,k in zip(rows,keep) if k]
    tp=sum((r['parent'],r['child']) in truth for r in selected)
    n=len(selected);precision=tp/n if n else 0.;recall=tp/len(truth) if truth else 0.
    return {'predicted':n,'tp':tp,'fp':n-tp,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,required=True)
    p.add_argument('--retrained',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=float,default=1.5,help='Predeclared calibration alert budget, default 1.5x')
    p.add_argument('--min-predictions',type=int,default=30)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory is nonempty; refusing overwrite')
    if a.budget<=0 or a.min_predictions<1:p.error('Invalid budget or min-predictions')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    import e5c38_end_to_end as c38
    with a.ranker.open('rb') as f:baseline=pickle.load(f)
    with a.retrained.open('rb') as f:proposed=pickle.load(f)
    with a.gate.open('rb') as f:old_gate=pickle.load(f)
    if baseline['features']!=base.FEATURES or proposed['features']!=base.FEATURES[:-1]:raise ValueError('Reranker feature schema mismatch')
    if old_gate['features']!=GATE_FEATURES:raise ValueError('Gate feature schema mismatch')
    if proposed.get('baseline_sha256')!=sha(a.ranker) or old_gate.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Model lineage mismatch')
    if float(baseline['threshold'])!=.65 or int(baseline['window'])!=90 or int(proposed['window'])!=90:raise ValueError('Frozen settings mismatch')
    for directory in (a.train,a.calibration):
        for name in ('telemetry.csv','ground_truth.csv'):
            if not (directory/name).is_file():p.error(f'Missing dataset file: {directory/name}')
    train_events=read(a.train/'telemetry.csv');cal_events=read(a.calibration/'telemetry.csv')
    train_truth=truth_edges(a.train/'ground_truth.csv');cal_truth=truth_edges(a.calibration/'ground_truth.csv')
    # Gate fit: only original E5-C15 training split, signal-selected (no fitted detector selection leakage).
    train_chosen={r['event_id'] for r in train_events if float(r['signal'])>=.65}
    train_rows=top_rows(train_events,train_chosen,proposed,90)
    y=np.asarray([int((r['parent'],r['child']) in train_truth) for r in train_rows],dtype=int)
    if len(train_rows)<30 or len(set(y.tolist()))!=2:raise RuntimeError('Insufficient positive/negative gate training examples')
    X=np.asarray([r['features'] for r in train_rows],dtype=float)
    gate_model=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=0)
    gate_model.fit(X,y)
    # Calibration: E5-C17 labels only. Detector fit on E5-C15 only.
    train_labels={r['event_id']:int(r['malicious']) for r in read(a.train/'ground_truth.csv')}
    ids=[r['event_id'] for r in train_events];feat=c38.extract(train_events,90)
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([train_labels[i] for i in ids]))
    cids=[r['event_id'] for r in cal_events];cfeat=c38.extract(cal_events,90)
    cp=detector.predict_proba(np.asarray([cfeat[i] for i in cids]))[:,1]
    alert_n=min(len(cids),int(round(sum(float(r['signal'])>=.65 for r in cal_events)*a.budget)))
    chosen={i for i,_ in sorted(zip(cids,cp),key=lambda z:(-float(z[1]),z[0]))[:alert_n]}
    new_rows=top_rows(cal_events,chosen,proposed,90)
    old_rows=top_rows(cal_events,chosen,baseline,90)
    if not new_rows or len(new_rows)!=len(old_rows):raise RuntimeError('Inconsistent candidate children')
    if [r['child'] for r in new_rows]!=[r['child'] for r in old_rows]:raise RuntimeError('Candidate child mismatch')
    probs=gate_model.predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1]
    old_probs=old_gate['model'].predict_proba(np.asarray([r['features'] for r in old_rows]))[:,1]
    # Threshold selection on E5-C17 only; no independent seeds accessed.
    curve=[]
    for cutoff in sorted(set(map(float,probs))):
        keep=probs>=cutoff
        if int(keep.sum())<a.min_predictions:continue
        curve.append({'cutoff':cutoff,**metrics(new_rows,cal_truth,keep)})
    if not curve:raise RuntimeError('No admissible calibration thresholds')
    best=max(curve,key=lambda r:(r['f1'],r['precision'],r['cutoff']))
    baseline_m=metrics(old_rows,cal_truth,old_probs>=old_gate['cutoff'])
    new_m=metrics(new_rows,cal_truth,probs>=best['cutoff'])
    a.output.mkdir(parents=True,exist_ok=True)
    model_obj={'model':gate_model,'features':GATE_FEATURES,'cutoff':best['cutoff'],'e5c56_sha256':sha(a.retrained),'baseline_e5c15_sha256':sha(a.ranker),'threshold':.65,'window':90,'budget':a.budget,'calibration_dataset':'e5c17_calibration'}
    with (a.output/'compatible_gate.pkl').open('wb') as f:pickle.dump(model_obj,f)
    write_csv(a.output/'calibration_curve.csv',curve)
    audit={'experiment':'E5-C58','status':'TRAINING + DEVELOPMENT CALIBRATION ONLY; no independent evaluation','train_gate_rows':len(train_rows),'train_correct_top1':int(y.sum()),'train_incorrect_top1':int(len(y)-y.sum()),'calibration_alerts':alert_n,'calibration_children':len(new_rows),'calibration_true_edges':len(cal_truth),'budget_multiplier':a.budget,'baseline':baseline_m,'proposed':new_m,'selected_cutoff':best['cutoff'],'min_predictions':a.min_predictions,'training_selection':'signal >=0.65; gate training labels from e5c15_train only','calibration_selection':'context detector fitted on training split; top N from E5-C17 telemetry','sha256':{'baseline_ranker':sha(a.ranker),'new_ranker':sha(a.retrained),'old_gate':sha(a.gate),'train_telemetry':sha(a.train/'telemetry.csv'),'train_truth':sha(a.train/'ground_truth.csv'),'cal_telemetry':sha(a.calibration/'telemetry.csv'),'cal_truth':sha(a.calibration/'ground_truth.csv')},'limitations':['Calibration F1 is in-sample and optimistic','Gate training signal-selected but calibration context-selected: distribution shift possible','Frozen original gate calibration may shift at budget 1.5x','Original reranker and new reranker can select different parents','Same synthetic generator, no real-world validation','No E5-C57 or other independent test seeds used']}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('E5-C58 | TRAIN-ONLY COMPATIBLE GATE + DEVELOPMENT CALIBRATION')
    print(f'Train: {len(train_rows)} candidate children; correct top-1={int(y.sum())}; incorrect={int(len(y)-y.sum())}')
    print(f'Calibration: alerts={alert_n} children={len(new_rows)} true_edges={len(cal_truth)} budget=x{a.budget:g}')
    print('Baseline:',json.dumps(baseline_m))
    print('Proposed:',json.dumps(new_m))
    print(f'New gate cutoff: {best["cutoff"]:.8f}')
    print('Saved:',a.output.resolve())
    print('Do NOT claim independent improvement; reserve new seeds for E5-C59.')

if __name__=='__main__':main()
