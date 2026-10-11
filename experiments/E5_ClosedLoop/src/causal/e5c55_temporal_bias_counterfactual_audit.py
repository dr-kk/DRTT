"""E5-C55: frozen-ranker temporal-feature counterfactual diagnostic.

Synthetic development seeds 5200-5204. No fitting or threshold optimization.
Counterfactuals modify inference-time features only; they are out-of-distribution
ablations, not deployable models or causal interventions on event time.
"""
import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS=range(5200,5205)
BUDGETS=(1.,1.25,1.5,2.,3.)
VARIANTS=('original','gap_neutral','gap_rank_neutral','both_neutral')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write_csv(path,rows):
    if not rows:raise ValueError('No rows for '+str(path))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ranked_candidates(events,chosen,ranker,window):
    ordered=sorted((e for e in events if e['event_id'] in chosen),key=lambda e:(int(e['timestamp']),e['event_id']))
    results={variant:{} for variant in VARIANTS}
    for j,child in enumerate(ordered):
        ct=int(child['timestamp']); entries=[]
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
                float(parent['sensor']==child['sensor']),float(parent['event_type']==child['event_type']),
                float(parent['signal']),float(child['signal']),len(entries)/50.,ix/50.])
        original=np.asarray(vectors,dtype=float)
        for variant in VARIANTS:
            X=original.copy()
            if variant in ('gap_neutral','both_neutral'):X[:,0]=0.5
            if variant in ('gap_rank_neutral','both_neutral'):X[:,10]=0.5
            scores=np.asarray(ranker['model'].decision_function(X)).reshape(-1)
            order=sorted(range(len(entries)),key=lambda i:(-float(scores[i]),entries[i][1],entries[i][0]['event_id']))
            results[variant][child['event_id']]=[entries[i][0]['event_id'] for i in order]
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--reference',type=Path,default=Path('results/causal/e5c53/per_seed.csv'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c55'))
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory not empty; preserving existing results')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c38_end_to_end as c38
    import e5c53_candidate_density_ranking_audit as c53
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Gate/ranker hash mismatch')
    if ranker['features']!=base.FEATURES:raise ValueError('Ranker schema mismatch')
    if float(ranker['threshold'])!=.65 or int(ranker['window'])!=90:raise ValueError('Unexpected frozen configuration')
    with a.reference.open(newline='',encoding='utf-8-sig') as f:
        refs={(int(r['seed']),float(r['budget_multiplier'])):r for r in csv.DictReader(f)}
    if len(refs)!=25:raise ValueError('Expected 25 E5-C53 reference rows')
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    feat=c38.extract(train,90); ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    rows=[]; cases=[]
    for seed in SEEDS:
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]; feat=c38.extract(events,90)
        probabilities=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
        alert_order=[i for i,_ in sorted(zip(ids,probabilities),key=lambda z:(-float(z[1]),z[0]))]
        base_budget=sum(float(r['signal'])>=.65 for r in events)
        true_parent={r['event_id']:r['predecessor_event_id'] for r in truthrows if r['predecessor_event_id']}
        for budget in BUDGETS:
            chosen=set(alert_order[:min(len(events),int(round(base_budget*budget)))])
            rankings=ranked_candidates(events,chosen,ranker,90)
            original=c53.candidate_ranks(events,chosen,ranker,90)
            if rankings['original']!=original:raise AssertionError(f'Original ranking mismatch seed={seed} x{budget}')
            eligible={child:parent for child,parent in true_parent.items() if parent in original.get(child,[])}
            ref=refs[(seed,budget)]
            if len(eligible)!=int(ref['eligible_truth_edges']):raise AssertionError('Eligibility mismatch with E5-C53')
            original_correct=sum(original[child][0]==parent for child,parent in eligible.items())
            if original_correct!=int(ref['top1_correct']):raise AssertionError('Original Top-1 mismatch with E5-C53')
            for variant in VARIANTS:
                ranks=rankings[variant]
                top1=sum(ranks[child][0]==parent for child,parent in eligible.items())
                mrr=float(np.mean([1/(ranks[child].index(parent)+1) for child,parent in eligible.items()])) if eligible else 0.
                gained=sum(original[child][0]!=parent and ranks[child][0]==parent for child,parent in eligible.items())
                lost=sum(original[child][0]==parent and ranks[child][0]!=parent for child,parent in eligible.items())
                rows.append({'seed':seed,'budget_multiplier':budget,'variant':variant,
                    'eligible_truth_edges':len(eligible),'top1_correct':top1,
                    'top1_accuracy':top1/len(eligible) if eligible else 0.,'mrr':mrr,
                    'recovered_from_original':gained,'new_errors_vs_original':lost,
                    'net_correct_change':top1-original_correct})
                if variant!='original':
                    for child,parent in eligible.items():
                        original_rank=original[child].index(parent)+1
                        new_rank=ranks[child].index(parent)+1
                        if original_rank!=new_rank or original[child][0]!=ranks[child][0]:
                            cases.append({'seed':seed,'budget_multiplier':budget,'variant':variant,
                                'child':child,'true_parent':parent,'candidate_count':len(ranks[child]),
                                'original_top1':original[child][0],'counterfactual_top1':ranks[child][0],
                                'original_true_rank':original_rank,'counterfactual_true_rank':new_rank,
                                'original_correct':int(original_rank==1),'counterfactual_correct':int(new_rank==1)})
            print(f'Seed {seed} x{budget:g}: eligible={len(eligible)} original={original_correct} '
                  + ' '.join(f'{v}={next(r["top1_correct"] for r in rows[-4:] if r["variant"]==v)}' for v in VARIANTS[1:]))
    summary=[]
    for budget in BUDGETS:
        for variant in VARIANTS:
            subset=[r for r in rows if r['budget_multiplier']==budget and r['variant']==variant]
            eligible=sum(r['eligible_truth_edges'] for r in subset)
            top1=sum(r['top1_correct'] for r in subset)
            summary.append({'budget_multiplier':budget,'variant':variant,'eligible_truth_edges':eligible,
                'top1_correct':top1,'top1_accuracy':top1/eligible if eligible else 0.,
                'mean_mrr':float(np.mean([r['mrr'] for r in subset])),
                'recovered_from_original':sum(r['recovered_from_original'] for r in subset),
                'new_errors_vs_original':sum(r['new_errors_vs_original'] for r in subset),
                'net_correct_change':sum(r['net_correct_change'] for r in subset)})
    report={'experiment':'E5-C55','seeds':list(SEEDS),'budgets':list(BUDGETS),
        'variants':{'original':'unchanged frozen inference',
            'gap_neutral':'replace gap feature with constant 0.5',
            'gap_rank_neutral':'replace gap_rank feature with constant 0.5',
            'both_neutral':'replace both gap and gap_rank features with constant 0.5'},
        'summary':summary,'sha256':{'ranker':sha(a.ranker),'gate':sha(a.gate),
            'reference':sha(a.reference),'train_telemetry':sha(a.train/'telemetry.csv')},
        'limitations':['Synthetic reused development seeds; no independent validation',
            'Counterfactual inference-feature ablations are out of training distribution',
            'No retraining or threshold selection',
            'Ground truth used only for post-inference diagnostics',
            'Only conditional ranking evaluated; no gate/F1 deployment claims',
            'These results cannot establish causality of temporal proximity']}
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'per_seed.csv',rows)
    write_csv(a.output/'budget_summary.csv',summary)
    if cases:write_csv(a.output/'changed_rank_cases.csv',cases)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C55 | TEMPORAL FEATURE COUNTERFACTUAL AUDIT')
    for r in summary:
        print(f'x{r["budget_multiplier"]:g} {r["variant"]}: top1={r["top1_correct"]}/{r["eligible_truth_edges"]} '
              f'acc={r["top1_accuracy"]:.3f} MRR={r["mean_mrr"]:.3f} '
              f'recovered={r["recovered_from_original"]} new_errors={r["new_errors_vs_original"]}')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
