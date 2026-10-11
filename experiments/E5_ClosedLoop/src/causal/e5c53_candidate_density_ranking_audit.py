"""E5-C53: diagnostic candidate-density and true-parent-rank audit.
Replays E5-C52 seeds/budgets with frozen detector/ranker/gate; no fitting or tuning.
Ground truth is used only for audit labels after inference. Synthetic data only.
"""
import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS=range(5200,5205)
BUDGETS=(1.,1.25,1.5,2.,3.)
BUCKETS=((1,1,'1'),(2,3,'2-3'),(4,7,'4-7'),(8,15,'8-15'),(16,10**9,'16+'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save_csv(path,rows):
    if not rows:raise ValueError(f'No rows: {path}')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def candidate_ranks(events,chosen,ranker,window):
    """Exact candidate features, scoring and tie-breaking from E5-C45."""
    ordered=sorted((r for r in events if r['event_id'] in chosen),
                   key=lambda r:(int(r['timestamp']),r['event_id']))
    result={}
    for j,child in enumerate(ordered):
        ct=int(child['timestamp']);entries=[]
        for parent in reversed(ordered[:j]):
            gap=ct-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or
                    parent['source_host']==child['source_host'] or
                    parent['destination_host']==child['destination_host']):continue
            entries.append((parent,gap))
        entries.sort(key=lambda z:(z[1],z[0]['event_id']))
        if not entries:continue
        vectors=[]
        for ix,(parent,gap) in enumerate(entries):
            vectors.append([gap/90.,float(parent['source_host']==child['source_host']),
                float(parent['destination_host']==child['destination_host']),
                float(parent['destination_host']==child['source_host']),
                float(parent['source_host']==child['destination_host']),
                float(parent['sensor']==child['sensor']),
                float(parent['event_type']==child['event_type']),
                float(parent['signal']),float(child['signal']),len(entries)/50.,ix/50.])
        scores=np.asarray(ranker['model'].decision_function(np.asarray(vectors,dtype=float))).reshape(-1)
        order=sorted(range(len(entries)),key=lambda i:(-float(scores[i]),entries[i][1],entries[i][0]['event_id']))
        result[child['event_id']]=[entries[i][0]['event_id'] for i in order]
    return result

def bucket(n):
    return next(label for low,high,label in BUCKETS if low<=n<=high)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--reference',type=Path,default=Path('results/causal/e5c52/per_seed.csv'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c53'))
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory not empty; preserving previous results')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c38_end_to_end as c38
    import e5c45_end_to_end_candidate_filter as c45
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Gate-ranker hash mismatch')
    if ranker['features']!=base.FEATURES or gate['features']!=c38.GATE_FEATURES:raise ValueError('Feature schema mismatch')
    if float(ranker['threshold'])!=.65 or int(ranker['window'])!=90 or int(gate['window'])!=90:raise ValueError('Frozen config mismatch')
    with a.reference.open(newline='',encoding='utf-8-sig') as f:
        refs={(int(r['seed']),float(r['budget_multiplier'])):r for r in csv.DictReader(f)}
    if len(refs)!=len(SEEDS)*len(BUDGETS):raise ValueError('Reference must have 25 unique seed-budget rows')
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    features=c38.extract(train,90);ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([features[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    rows=[];detail=[];bucket_rows=[]
    for seed in SEEDS:
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        feat=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
        ranked=[i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))]
        base_budget=sum(float(r['signal'])>=.65 for r in events)
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        truth_parent={child:parent for parent,child in truth}
        for multiplier in BUDGETS:
            budget=min(len(events),int(round(base_budget*multiplier)))
            chosen=set(ranked[:budget])
            ranked_candidates=candidate_ranks(events,chosen,ranker,90)
            records=c45.candidates_with_features(events,chosen,ranker,90)
            if len(records)!=len(ranked_candidates):raise AssertionError('Candidate child count mismatch')
            top1={(r['parent'],r['child']) for r in records}
            if top1!={(parents[0],child) for child,parents in ranked_candidates.items()}:
                raise AssertionError('Ranker top1 differs from E5-C45')
            gp=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records],dtype=float))[:,1] if records else []
            accepted={(r['parent'],r['child']) for r,pr in zip(records,gp) if pr>=gate['cutoff']}
            metrics=c45.evaluate(accepted,truth,len(events))
            ref=refs[(seed,multiplier)]
            for k in ('tp','fp','predicted','top1_correct','candidate_children','both_alerted','true_parent_candidate'):
                value={'top1_correct':len(top1&truth),'candidate_children':len(records),
                       'both_alerted':sum(p in chosen and c in chosen for p,c in truth),
                       'true_parent_candidate':sum(p in ranked_candidates.get(c,[]) for p,c in truth)}.get(k,metrics.get(k))
                if int(ref[k])!=value:raise AssertionError(f'Seed {seed} x{multiplier}: {k} {value} != E5-C52 {ref[k]}')
            counts=[len(v) for v in ranked_candidates.values()]
            eligible=[]
            for child,parents in ranked_candidates.items():
                true_parent=truth_parent.get(child)
                if true_parent not in parents:continue
                true_rank=parents.index(true_parent)+1
                eligible.append((child,len(parents),true_rank))
                detail.append({'seed':seed,'budget_multiplier':multiplier,'child':child,
                               'candidate_count':len(parents),'candidate_bucket':bucket(len(parents)),
                               'true_parent_rank':true_rank,'top1_correct':int(true_rank==1),
                               'reciprocal_rank':1/true_rank})
            if len(eligible)!=int(ref['true_parent_candidate']):raise AssertionError('Eligible truth mismatch')
            rr={'seed':seed,'budget_multiplier':multiplier,'alerts':budget,
                'candidate_children':len(counts),'candidate_pairs':sum(counts),
                'mean_candidates_per_child':float(np.mean(counts)) if counts else 0,
                'median_candidates_per_child':float(np.median(counts)) if counts else 0,
                'p90_candidates_per_child':float(np.quantile(counts,.9)) if counts else 0,
                'max_candidates_per_child':max(counts,default=0),
                'eligible_truth_edges':len(eligible),
                'top1_correct':sum(r==1 for _,_,r in eligible),
                'top1_accuracy':float(np.mean([r==1 for _,_,r in eligible])) if eligible else 0,
                'mean_reciprocal_rank':float(np.mean([1/r for _,_,r in eligible])) if eligible else 0,
                'mean_true_parent_rank':float(np.mean([r for _,_,r in eligible])) if eligible else 0,
                'top3_recall_given_candidate':float(np.mean([r<=3 for _,_,r in eligible])) if eligible else 0,
                'tp':metrics['tp'],'fp':metrics['fp'],'f1':metrics['f1']}
            rows.append(rr)
            for lo,hi,name in BUCKETS:
                group=[r for _,n,r in eligible if lo<=n<=hi]
                bucket_rows.append({'seed':seed,'budget_multiplier':multiplier,'candidate_bucket':name,
                    'eligible_truth_edges':len(group),'top1_correct':sum(r==1 for r in group),
                    'top1_accuracy':sum(r==1 for r in group)/len(group) if group else None,
                    'mean_reciprocal_rank':float(np.mean([1/r for r in group])) if group else None})
            print(f'Seed {seed} x{multiplier:g}: candidates/child={rr["mean_candidates_per_child"]:.2f} '
                  f'eligible={len(eligible)} top1={rr["top1_correct"]} '
                  f'top1_acc={rr["top1_accuracy"]:.3f} MRR={rr["mean_reciprocal_rank"]:.3f}')
    summary=[]
    for mult in BUDGETS:
        subset=[r for r in rows if r['budget_multiplier']==mult]
        d=[r for r in detail if r['budget_multiplier']==mult]
        summary.append({'budget_multiplier':mult,
            'mean_candidates_per_child':float(np.mean([r['mean_candidates_per_child'] for r in subset])),
            'candidate_pairs_total':sum(r['candidate_pairs'] for r in subset),
            'eligible_truth_edges':len(d),
            'top1_correct':sum(r['top1_correct'] for r in d),
            'top1_accuracy':float(np.mean([r['top1_correct'] for r in d])) if d else 0,
            'mean_reciprocal_rank':float(np.mean([r['reciprocal_rank'] for r in d])) if d else 0,
            'top3_recall_given_candidate':float(np.mean([r['true_parent_rank']<=3 for r in d])) if d else 0,
            'tp':sum(r['tp'] for r in subset),'fp':sum(r['fp'] for r in subset),
            'f1_mean':float(np.mean([r['f1'] for r in subset]))})
    report={'experiment':'E5-C53','scope':'Post-hoc same-generator development-seed audit of E5-C52, no training or threshold tuning',
        'seeds':list(SEEDS),'budgets':list(BUDGETS),'summary':summary,
        'sha256':{'ranker':sha(a.ranker),'gate':sha(a.gate),'reference':sha(a.reference),'train_telemetry':sha(a.train/'telemetry.csv')},
        'limitations':['Same synthetic generator; not independent validation',
          'True parent rank requires ground truth and is used only for diagnosis',
          'Bucket comparisons are descriptive and confounded by alert-budget composition',
          'Candidate observations within a seed are dependent; no event-level confidence intervals',
          'Do not select an operating point using this post-hoc analysis']}
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/'per_seed.csv',rows)
    save_csv(a.output/'eligible_edge_ranks.csv',detail)
    save_csv(a.output/'bucket_per_seed.csv',bucket_rows)
    save_csv(a.output/'budget_summary.csv',summary)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C53 | CANDIDATE DENSITY AND RANKING AUDIT')
    for s in summary:
        print(f'x{s["budget_multiplier"]:g}: candidates/child={s["mean_candidates_per_child"]:.2f} '
              f'eligible={s["eligible_truth_edges"]} top1={s["top1_correct"]} '
              f'top1_acc={s["top1_accuracy"]:.3f} MRR={s["mean_reciprocal_rank"]:.3f} '
              f'TP/FP={s["tp"]}/{s["fp"]}')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
