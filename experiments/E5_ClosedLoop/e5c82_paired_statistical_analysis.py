"""E5-C82: paired, seed-cluster statistical analysis of frozen E5-C81 outputs.
No model training, no test-time model selection, and no overwrite of previous results.
Run from D:\\DRTT\\E5_ClosedLoop.
"""
import csv, json, hashlib, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path.cwd()
SRC=ROOT/'results/causal/e5c81_fair_baseline_comparison/test_seed_metrics.csv'
OUT=ROOT/'results/causal/e5c82_paired_statistical_analysis'
METHODS=['Frozen','HardNegative','RecentParent','HostHandoff','TemporalHost','SeededRandom']
COMPARISONS=[('HardNegative','Frozen'),('HardNegative','TemporalHost'),('Frozen','TemporalHost'),('HardNegative','SeededRandom')]
VARIANTS=['original','matched_signal']
BOOT=20000
SEED=820082

def dump(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def f1(tp,pred,truth):
    return 2.0*tp/(pred+truth) if pred+truth else 0.0

def pooled(arr):
    return f1(sum(int(r['tp']) for r in arr),sum(int(r['predicted']) for r in arr),sum(int(r['truth']) for r in arr))

def signflip_p(diff):
    # Exact two-sided sign-flip randomization of paired differences (2^n sign patterns).
    observed=abs(float(np.mean(diff)))
    dist=[]
    for signs in itertools.product((-1.,1.),repeat=len(diff)):
        dist.append(abs(float(np.mean(diff*np.asarray(signs)))))
    return float(sum(x>=observed-1e-12 for x in dist)/len(dist))

def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
    if not SRC.is_file():raise FileNotFoundError(f'Missing E5-C81 per-seed data: {SRC}')
    with SRC.open(newline='',encoding='utf-8-sig') as f: data=list(csv.DictReader(f))
    if not data:raise RuntimeError('Empty input')
    lookup={}
    for r in data:
        key=(r['variant'],int(r['seed']),r['method'])
        if key in lookup:raise RuntimeError(f'Duplicate row {key}')
        lookup[key]=r
        for k in ('tp','fp','fn','truth','predicted'):
            if int(r[k])<0:raise RuntimeError(f'Negative {k} at {key}')
        if int(r['tp'])+int(r['fp'])!=int(r['predicted']) or int(r['tp'])+int(r['fn'])!=int(r['truth']):
            raise RuntimeError(f'Inconsistent counts at {key}')
    seeds=sorted({k[1] for k in lookup})
    if len(seeds)!=5 or seeds!=list(range(8101,8106)):
        raise RuntimeError(f'Expected untouched E5-C81 seeds 8101..8105; got {seeds}')
    for v in VARIANTS:
        for seed in seeds:
            for method in METHODS:
                if (v,seed,method) not in lookup:raise RuntimeError(f'Missing {(v,seed,method)}')
            counts={(int(lookup[(v,seed,m)]['truth']),int(lookup[(v,seed,m)]['predicted'])) for m in METHODS}
            if len(counts)!=1:raise RuntimeError(f'Unequal truth/prediction budgets: {v} seed {seed}: {counts}')
    rng=np.random.default_rng(SEED)
    indices=rng.integers(0,len(seeds),size=(BOOT,len(seeds)))
    method_rows=[];pair_rows=[];seed_rows=[]
    for v in VARIANTS:
        for m in METHODS:
            group=[lookup[(v,s,m)] for s in seeds]
            tp=np.array([int(r['tp']) for r in group],dtype=float)
            pr=np.array([int(r['predicted']) for r in group],dtype=float)
            tr=np.array([int(r['truth']) for r in group],dtype=float)
            boot_tp=tp[indices].sum(axis=1)
            boot_den=(pr[indices]+tr[indices]).sum(axis=1)
            boot_f1=np.divide(2*boot_tp,boot_den,out=np.zeros_like(boot_tp),where=boot_den>0)
            lo,hi=np.quantile(boot_f1,[.025,.975])
            method_rows.append(dict(variant=v,method=m,seeds=len(seeds),truth=int(tr.sum()),predicted=int(pr.sum()),tp=int(tp.sum()),pooled_f1=pooled(group),bootstrap_ci95_low=float(lo),bootstrap_ci95_high=float(hi)))
            for s,r in zip(seeds,group):seed_rows.append(dict(variant=v,seed=s,method=m,tp=int(r['tp']),predicted=int(r['predicted']),truth=int(r['truth']),f1=f1(int(r['tp']),int(r['predicted']),int(r['truth']))))
        for a,b in COMPARISONS:
            A=[lookup[(v,s,a)] for s in seeds];B=[lookup[(v,s,b)] for s in seeds]
            tp_a=np.array([int(r['tp']) for r in A],dtype=float)
            tp_b=np.array([int(r['tp']) for r in B],dtype=float)
            den_a=np.array([int(r['predicted'])+int(r['truth']) for r in A],dtype=float)
            den_b=np.array([int(r['predicted'])+int(r['truth']) for r in B],dtype=float)
            delta=2*tp_a.sum()/den_a.sum()-2*tp_b.sum()/den_b.sum()
            ba=2*tp_a[indices].sum(axis=1)/den_a[indices].sum(axis=1)
            bb=2*tp_b[indices].sum(axis=1)/den_b[indices].sum(axis=1)
            lo,hi=np.quantile(ba-bb,[.025,.975])
            per_seed=2*tp_a/den_a-2*tp_b/den_b
            pair_rows.append(dict(variant=v,method_a=a,method_b=b,seeds=len(seeds),f1_a=pooled(A),f1_b=pooled(B),delta_pooled_f1=float(delta),paired_bootstrap_ci95_low=float(lo),paired_bootstrap_ci95_high=float(hi),exact_signflip_p_two_sided=signflip_p(per_seed),mean_seed_delta=float(per_seed.mean()),positive_seed_differences=int((per_seed>0).sum()),negative_seed_differences=int((per_seed<0).sum()),zero_seed_differences=int((per_seed==0).sum())))
    # Holm family-wise correction across all eight predeclared comparisons.
    order=sorted(range(len(pair_rows)),key=lambda i:pair_rows[i]['exact_signflip_p_two_sided'])
    adjusted=[0.]*len(pair_rows);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*pair_rows[i]['exact_signflip_p_two_sided']))
        adjusted[i]=running
    for r,p in zip(pair_rows,adjusted):r['holm_adjusted_p']=p
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'method_ci95.csv',method_rows)
    dump(OUT/'paired_comparisons.csv',pair_rows)
    dump(OUT/'seed_f1.csv',seed_rows)
    (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C82',input=str(SRC.relative_to(ROOT)),input_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),seeds=seeds,bootstrap_replicates=BOOT,bootstrap_random_seed=SEED,comparison_family=[list(x) for x in COMPARISONS],variants=VARIANTS,bootstrap_unit='Independent generator seed; paired cluster resampling across methods',ci='Percentile paired bootstrap 95%; exploratory with only five clusters',test='Exact two-sided paired sign-flip test on per-seed F1 differences',multiplicity='Holm correction across eight comparisons',limitations=['Only five independent synthetic seeds; bootstrap CIs may be unstable','Exact sign-flip test assumes exchangeability/symmetric null; with five seeds its p-value resolution is coarse','A small p-value does not establish real-world generalization','Heuristic baselines are not published SOTA methods','Test set used for evaluation only; no hyperparameter selection','Matched-signal perturbation is synthetic and label-informed during generation']),indent=2),encoding='utf-8')
    print('\nE5-C82 PAIRED COMPARISONS')
    for r in pair_rows:print(r)
    print('\nSaved:',OUT)

if __name__=='__main__':main()
