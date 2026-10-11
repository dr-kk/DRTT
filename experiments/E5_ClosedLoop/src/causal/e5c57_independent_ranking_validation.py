"""E5-C57: independent paired ranking evaluation on untouched synthetic seeds.

Uses E5-C56 frozen no-gap-rank model and original E5-C15 model.
No fitting, calibration, or operating-point selection on evaluation seeds.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS=range(5700,5710)
BUDGETS=(1.,1.25,1.5,2.,3.)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def save_csv(path,rows):
    if not rows:raise ValueError('No rows: '+str(path))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ci_seed_paired(differences):
    """95% percentile bootstrap CI of mean paired seed differences (fixed RNG)."""
    x=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(570057)
    draws=rng.choice(x,size=(20000,len(x)),replace=True).mean(axis=1)
    return float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker',type=Path,required=True)
    ap.add_argument('--retrained',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error('Output directory is not empty; refusing overwrite')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    import e5c38_end_to_end as c38
    import e5c5_overlapping_generator as gen
    import e5c56_temporal_feature_retraining as c56
    with a.ranker.open('rb') as f: original=pickle.load(f)
    with a.retrained.open('rb') as f: proposed=pickle.load(f)
    if original['features']!=base.FEATURES:raise ValueError('Original ranker feature schema mismatch')
    if original['features'][-1]!='gap_rank':raise ValueError('Unexpected original last feature')
    if proposed['features']!=base.FEATURES[:-1]:raise ValueError('Proposed ranker feature schema mismatch')
    if proposed['baseline_sha256']!=sha(a.ranker):raise ValueError('Proposed model trained against a different baseline')
    if float(original['threshold'])!=.65 or int(original['window'])!=90:raise ValueError('Unexpected original frozen settings')
    if proposed['threshold']!=original['threshold'] or proposed['window']!=original['window']:raise ValueError('Model settings mismatch')
    telemetry=a.train/'telemetry.csv';truthfile=a.train/'ground_truth.csv'
    train_events=c38.read(telemetry)
    labels={r['event_id']:int(r['malicious']) for r in c38.read(truthfile)}
    ids=[r['event_id'] for r in train_events]
    train_features=c38.extract(train_events,90)
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([train_features[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    rows=[];details=[]
    for seed in SEEDS:
        events,truth=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        feat=c38.extract(events,90)
        probabilities=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
        order=[i for i,_ in sorted(zip(ids,probabilities),key=lambda z:(-float(z[1]),z[0]))]
        base_n=sum(float(e['signal'])>=.65 for e in events)
        true_parent={r['event_id']:r['predecessor_event_id'] for r in truth if r['predecessor_event_id']}
        if len(true_parent)!=600:raise AssertionError(f'Seed {seed}: expected 600 true edges')
        for budget in BUDGETS:
            n=min(len(events),int(round(base_n*budget)))
            chosen=set(order[:n])
            ranks=c56.rank_candidates(events,chosen,original,proposed['model'],90)
            if set(ranks)!= {'original','no_gap_rank'}:raise AssertionError('Unexpected ranking variants')
            if set(ranks['original'])!=set(ranks['no_gap_rank']):raise AssertionError('Candidate child sets differ')
            eligible={c:p for c,p in true_parent.items() if p in ranks['original'].get(c,[])}
            if not eligible:raise AssertionError('No eligible edges')
            per={}
            for name in ('original','no_gap_rank'):
                correct=0; reciprocal=0.
                for child,parent in eligible.items():
                    ranking=ranks[name][child]
                    if parent not in ranking:raise AssertionError('True parent lost from a ranking')
                    correct+=int(ranking[0]==parent)
                    reciprocal+=1./(ranking.index(parent)+1)
                per[name]=(correct,reciprocal/len(eligible))
                rows.append(dict(seed=seed,budget_multiplier=budget,variant=name,selected_alerts=n,eligible=len(eligible),top1_correct=correct,top1_accuracy=correct/len(eligible),mrr=reciprocal/len(eligible)))
            old, new=per['original'],per['no_gap_rank']
            details.append(dict(seed=seed,budget_multiplier=budget,eligible=len(eligible),baseline_correct=old[0],proposed_correct=new[0],delta_correct=new[0]-old[0],delta_accuracy=(new[0]-old[0])/len(eligible),delta_mrr=new[1]-old[1]))
            print(f'Seed {seed} x{budget:g}: eligible={len(eligible)} original={old[0]} proposed={new[0]} delta={new[0]-old[0]:+d}')
    summaries=[]
    for budget in BUDGETS:
        g=[r for r in details if r['budget_multiplier']==budget]
        eligible=sum(r['eligible'] for r in g)
        old=sum(r['baseline_correct'] for r in g)
        new=sum(r['proposed_correct'] for r in g)
        ds=[r['delta_accuracy'] for r in g]
        ci=ci_seed_paired(ds)
        mrrds=[r['delta_mrr'] for r in g]
        mrrci=ci_seed_paired(mrrds)
        summaries.append(dict(budget_multiplier=budget,seeds=len(g),eligible=eligible,baseline_correct=old,proposed_correct=new,baseline_top1=old/eligible,proposed_top1=new/eligible,pooled_delta_pp=100*(new-old)/eligible,mean_seed_delta_pp=100*np.mean(ds),seed_bootstrap_95ci_low_pp=100*ci[0],seed_bootstrap_95ci_high_pp=100*ci[1],improved_seeds=sum(r['delta_correct']>0 for r in g),tied_seeds=sum(r['delta_correct']==0 for r in g),worsened_seeds=sum(r['delta_correct']<0 for r in g),mean_seed_delta_mrr=float(np.mean(mrrds)),mrr_seed_bootstrap_95ci_low=mrrci[0],mrr_seed_bootstrap_95ci_high=mrrci[1]))
        print(f'x{budget:g}: baseline={old}/{eligible} ({old/eligible:.4f}), proposed={new}/{eligible} ({new/eligible:.4f}), delta={100*(new-old)/eligible:+.2f} pp, seed CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp, seeds +/0/-={summaries[-1]["improved_seeds"]}/{summaries[-1]["tied_seeds"]}/{summaries[-1]["worsened_seeds"]}')
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/'per_seed.csv',rows)
    save_csv(a.output/'paired_seed_differences.csv',details)
    save_csv(a.output/'budget_summary.csv',summaries)
    audit=dict(experiment='E5-C57',purpose='Independent ranking evaluation of frozen E5-C56 model',seeds=list(SEEDS),budgets=list(BUDGETS),original_sha256=sha(a.ranker),proposed_sha256=sha(a.retrained),training_telemetry_sha256=sha(telemetry),training_truth_sha256=sha(truthfile),model_fitting_on_evaluation_seeds=False,gate_evaluated=False,interpretation='Synthetic independent-seed ranking validation only; no end-to-end acceptance claims. Budget x1.5 was development best F1, not independently optimized here.',bootstrap='20,000 seed-level paired resamples; descriptive percentile confidence intervals; repeated budgets within seeds are correlated')
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
