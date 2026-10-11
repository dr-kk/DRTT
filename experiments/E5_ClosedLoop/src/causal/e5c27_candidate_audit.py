"""E5-C27: offline candidate/reranker audit on existing synthetic datasets.
No training, no model/calibration changes. Truth is read only AFTER inference.
The threshold sweep is diagnostic and must not be used to tune on held-out seeds.
"""
import argparse, csv, hashlib, importlib.util, json, pickle, statistics
from pathlib import Path
import numpy as np

FEATURES=['gap','same_source','same_destination','dst_to_src','src_to_dst','same_sensor','same_type','parent_signal','child_signal','candidate_count','gap_rank']

def read(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def div(a,b): return a/b if b else None

def run_one(folder,threshold,window,mod,obj):
    telemetry=folder/'telemetry.csv'
    events, selected, cand=mod.candidates(str(telemetry),threshold,window)
    # No truth available in this section: rank candidates using frozen model.
    selected_ids={r['event_id'] for r in selected}
    inferred={}
    total_candidates=0
    for child,entries in cand.items():
        total_candidates+=len(entries)
        feats=np.asarray([v for _,_,v in entries],dtype=float)
        scores=obj['model'].decision_function(feats)
        ranked=sorted(zip(entries,scores),key=lambda z:(-float(z[1]),z[0][1],z[0][0]))
        inferred[child]=ranked[0][0][0]
    candidate_ids={child:{parent for parent,_,_ in entries} for child,entries in cand.items()}
    # Truth loaded only after candidate generation and inference are finished.
    truth_rows=read(folder/'ground_truth.csv')
    truth={(r['predecessor_event_id'],r['event_id']) for r in truth_rows if r.get('predecessor_event_id')}
    event_by_id={r['event_id']:r for r in events}
    if any(p not in event_by_id or c not in event_by_id for p,c in truth):
        raise ValueError(f'Ground truth refers to unknown telemetry event IDs: {folder}')
    detected_truth={(p,c) for p,c in truth if p in selected_ids and c in selected_ids}
    window_truth={(p,c) for p,c in detected_truth if 0 < int(event_by_id[c]['timestamp'])-int(event_by_id[p]['timestamp']) <= window}
    candidate_truth={(p,c) for p,c in window_truth if p in candidate_ids.get(c,set())}
    predicted={(p,c) for c,p in inferred.items()}
    tp=len(predicted & truth)
    # Ranking loss: candidate-present ground truth not chosen as top-1.
    ranked_correct=len(predicted & candidate_truth)
    result={
        'dataset':folder.name,'threshold':threshold,'window':window,
        'events':len(events),'detected_events':len(selected),
        'truth_edges':len(truth),'detected_truth_edges':len(detected_truth),
        'window_eligible_truth_edges':len(window_truth),
        'candidate_eligible_truth_edges':len(candidate_truth),
        'detection_lost_edges':len(truth)-len(detected_truth),
        'window_lost_edges':len(detected_truth)-len(window_truth),
        'host_candidate_lost_edges':len(window_truth)-len(candidate_truth),
        'candidate_children':len(cand),'candidate_pairs':total_candidates,
        'candidate_recall_all_truth':div(len(candidate_truth),len(truth)),
        'candidate_recall_window_eligible':div(len(candidate_truth),len(window_truth)),
        'predicted_edges':len(predicted),'true_positive_edges':tp,
        'ranking_missed_candidate_edges':len(candidate_truth)-ranked_correct,
        'precision':div(tp,len(predicted)),'recall':div(tp,len(truth)),
        'f1':div(2*tp,len(predicted)+len(truth)),
        'false_edges_per_1000_events':div(1000*(len(predicted)-tp),len(events)),
        'candidates_per_selected_event':div(total_candidates,len(selected)),
        'telemetry_sha256':hashlib.sha256(telemetry.read_bytes()).hexdigest()
    }
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--scripts',required=True,help='Folder containing e5c15_reranker.py')
    p.add_argument('--model',required=True,help='Trusted, locally trained E5-C15 pickle')
    p.add_argument('--output',required=True)
    p.add_argument('--thresholds',nargs='+',type=float,default=[0.3,0.4,0.5,0.6,0.65,0.7,0.8])
    p.add_argument('--window',type=int,default=90)
    a=p.parse_args()
    src=Path(a.scripts)/'e5c15_reranker.py'
    spec=importlib.util.spec_from_file_location('e5c15_local',src)
    if not spec or not spec.loader:raise FileNotFoundError(src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # Pickle can execute arbitrary code: only load locally trained, trusted models.
    with open(a.model,'rb') as f:obj=pickle.load(f)
    if obj.get('features')!=FEATURES:raise ValueError('Frozen model feature schema mismatch')
    if a.window != int(obj['window']):
        raise ValueError('Window differs from trained model; rerun with model window')
    rows=[]
    for d in a.datasets:
        folder=Path(d)
        for t in a.thresholds:
            print(f'Running {folder.name}, threshold={t:g}',flush=True)
            rows.append(run_one(folder,t,a.window,mod,obj))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'per_dataset_threshold.csv',rows)
    agg=[]
    for t in a.thresholds:
        subset=[r for r in rows if r['threshold']==t]
        keys=['events','detected_events','truth_edges','detected_truth_edges','window_eligible_truth_edges',
              'candidate_eligible_truth_edges','detection_lost_edges','window_lost_edges',
              'host_candidate_lost_edges','candidate_children','candidate_pairs','predicted_edges',
              'true_positive_edges','ranking_missed_candidate_edges']
        s={'threshold':t,'datasets':len(subset)}
        s.update({k:sum(r[k] for r in subset) for k in keys})
        tp=s['true_positive_edges'];pred=s['predicted_edges'];true=s['truth_edges']
        s.update({'candidate_recall_all_truth':div(s['candidate_eligible_truth_edges'],true),
                  'candidate_recall_window_eligible':div(s['candidate_eligible_truth_edges'],s['window_eligible_truth_edges']),
                  'precision':div(tp,pred),'recall':div(tp,true),'f1':div(2*tp,pred+true),
                  'false_edges_per_1000_events':div(1000*(pred-tp),s['events']),
                  'candidates_per_selected_event':div(s['candidate_pairs'],s['detected_events'])})
        agg.append(s)
    save_csv(output/'threshold_aggregate.csv',agg)
    summary={'design':'Frozen model; candidate and top-1 ranking audit on synthetic E5-C18 telemetry',
             'model_path':str(Path(a.model)),
             'trained_threshold':obj.get('threshold'),'trained_window':obj.get('window'),
             'model_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
             'datasets':len(a.datasets),'thresholds':a.thresholds,
             'frozen_threshold_result':next((r for r in agg if abs(r['threshold']-float(obj['threshold']))<1e-9),None),
             'threshold_aggregate':agg,
             'limitations':[
                 'Thresholds other than the trained threshold change the input distribution; results are diagnostic, not validated operating points',
                 'Ground truth is read only after telemetry-only candidate generation and frozen ranking',
                 'No E5-C17 rejection applied: rejection code/calibration schema not yet inspected',
                 'Host-constraint and ranking decomposition applies to E5-C15 candidate generator only',
                 'All evaluation data are synthetic; do not tune on previously held-out E5-C18 seeds',
                 'Candidate eligibility is not proof of causal identifiability']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C27 | FROZEN CANDIDATE AND RANKING AUDIT')
    print(json.dumps({'frozen_threshold_result':summary['frozen_threshold_result'],'saved':str(output)},indent=2))

if __name__=='__main__':main()
