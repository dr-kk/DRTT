"""E5-C30: Frozen causal-link error diagnosis on DEVELOPMENT calibration data.
No training, no threshold optimization, no test-set access.
Requires trusted local e5c15_reranker.py and model pickle.
"""
import argparse, csv, importlib.util, json, pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def read_csv(p):
    with open(p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_ratio(a,b):
    return a/b if b else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--output', required=True)
    a=ap.parse_args()
    d=Path(a.dataset); out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    spec=importlib.util.spec_from_file_location('e5c15_reranker',Path(a.scripts)/'e5c15_reranker.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with open(a.model,'rb') as f: model=pickle.load(f)
    if model['features'] != module.FEATURES: raise ValueError('Model feature mismatch')
    cal=json.loads(Path(a.calibration).read_text(encoding='utf-8'))
    if cal['method']!='e5c17_score_rejection_v1':raise ValueError('Unexpected calibration format')
    cutoff=float(cal['selected_cutoff'])
    truth=read_csv(d/'ground_truth.csv')
    truth_by_id={r['event_id']:r for r in truth}
    if len(truth_by_id)!=len(truth):raise ValueError('Duplicate ground-truth event IDs')
    events, selected, cand=module.candidates(d/'telemetry.csv',model['threshold'],model['window'])
    event_by_id={r['event_id']:r for r in events}
    selected_ids={r['event_id'] for r in selected}
    true_edges={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    if not true_edges:raise ValueError('No ground truth edges')
    missing_truth_ids=set(truth_by_id)^set(event_by_id)
    if missing_truth_ids:raise ValueError('Telemetry/truth ID mismatch')
    predicted=[]; breakdown=Counter(); gap_stats=defaultdict(lambda:Counter())
    category_examples=defaultdict(list)
    candidate_truth=set(); rank_correct=0
    for child,entries in cand.items():
        if not entries:continue
        feats=np.asarray([v for _,_,v in entries],dtype=float)
        scores=model['model'].decision_function(feats)
        ranked=sorted(zip(entries,scores),key=lambda z:(-float(z[1]),z[0][1],z[0][0]))
        (parent,gap,_),score=ranked[0]
        truth_parent=truth_by_id[child]['predecessor_event_id']
        has_true=bool(truth_parent)
        truth_available=has_true and any(p==truth_parent for p,_,_ in entries)
        if truth_available:candidate_truth.add((truth_parent,child))
        correct=has_true and parent==truth_parent
        if correct:rank_correct+=1
        if not has_true:category='no_true_parent'
        elif not truth_available:category='true_parent_not_candidate'
        elif correct:category='correct_parent'
        else:category='wrong_parent_with_true_candidate'
        kept=float(score)>=cutoff
        breakdown[(category,'before')]+=1
        if kept:breakdown[(category,'after')]+=1
        pe=event_by_id[parent]; ce=event_by_id[child]
        relation='dst_to_src' if pe['destination_host']==ce['source_host'] else ('same_source' if pe['source_host']==ce['source_host'] else ('same_destination' if pe['destination_host']==ce['destination_host'] else 'other'))
        bucket='0-2' if gap<=2 else ('3-10' if gap<=10 else ('11-30' if gap<=30 else '31-90'))
        gap_stats[(bucket,relation)]['total']+=1
        gap_stats[(bucket,relation)]['correct']+=int(correct)
        gap_stats[(bucket,relation)]['retained']+=int(kept)
        gap_stats[(bucket,relation)]['retained_correct']+=int(kept and correct)
        if len(category_examples[category])<5:
            category_examples[category].append({'child':child,'predicted_parent':parent,'true_parent':truth_parent,'score':float(score),'retained':kept})
        predicted.append({'child_event_id':child,'predicted_parent':parent,'true_parent':truth_parent,'category':category,'score':float(score),'retained':int(kept),'gap':gap,'relation':relation,'candidate_count':len(entries)})
    # For each true edge, distinguish lost endpoint vs time window vs host rule vs ranking.
    losses=Counter()
    for parent,child in true_edges:
        if parent not in selected_ids or child not in selected_ids:
            losses['detection_lost']+=1
        elif (parent,child) not in candidate_truth:
            delta=int(event_by_id[child]['timestamp'])-int(event_by_id[parent]['timestamp'])
            if not 0<delta<=int(model['window']):losses['time_window_lost']+=1
            else:losses['candidate_rule_lost']+=1
        elif any(r['child_event_id']==child and r['predicted_parent']==parent for r in predicted):
            losses['correctly_ranked']+=1
        else:losses['ranking_lost']+=1
    if sum(losses.values())!=len(true_edges):raise AssertionError('True-edge partition does not sum to truth edges')
    before_tp=sum(r['category']=='correct_parent' for r in predicted)
    after=[r for r in predicted if r['retained']]
    after_tp=sum(r['category']=='correct_parent' for r in after)
    summary={
        'design':'Calibration-only frozen-model error diagnosis; descriptive, not external validation',
        'dataset':str(d),'events':len(events),'truth_edges':len(true_edges),
        'selected_events':len(selected),'candidate_children':len(predicted),
        'threshold':model['threshold'],'window':model['window'],'frozen_cutoff':cutoff,
        'true_edge_loss_partition':dict(losses),
        'predicted_child_categories':{k:{'before':breakdown[(k,'before')],'after':breakdown[(k,'after')]} for k in ['no_true_parent','true_parent_not_candidate','correct_parent','wrong_parent_with_true_candidate']},
        'before':{'predicted':len(predicted),'tp':before_tp,'fp':len(predicted)-before_tp,'precision':safe_ratio(before_tp,len(predicted)),'recall':safe_ratio(before_tp,len(true_edges))},
        'after':{'predicted':len(after),'tp':after_tp,'fp':len(after)-after_tp,'precision':safe_ratio(after_tp,len(after)),'recall':safe_ratio(after_tp,len(true_edges))},
        'example_errors':dict(category_examples),
        'limitations':['Ground truth used only for offline diagnostic labels','Calibration dataset previously used to choose cutoff','Synthetic campaign IDs are not causal proof','No tuning on evaluated E5-C18 test seeds']
    }
    write_csv(out/'predictions_diagnostic.csv',predicted,['child_event_id','predicted_parent','true_parent','category','score','retained','gap','relation','candidate_count'])
    rel_rows=[{'gap_bucket':k[0],'host_relation':k[1],**dict(v)} for k,v in sorted(gap_stats.items())]
    write_csv(out/'gap_host_breakdown.csv',rel_rows,['gap_bucket','host_relation','total','correct','retained','retained_correct'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C30 | FROZEN DEVELOPMENT ERROR DIAGNOSIS')
    print(json.dumps({k:summary[k] for k in ['events','truth_edges','candidate_children','true_edge_loss_partition','predicted_child_categories','before','after']},indent=2))
    print('Saved:',out.resolve())

if __name__=='__main__':main()
