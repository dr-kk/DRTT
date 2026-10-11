"""E5-C68: development-only cost-sensitive comparison of frozen E5-C67 policies.

Utility = -(C_FP * FP + C_FN * FN).  Sweep C_FP/C_FN; no model tuning.
"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='E5-C67 results directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fp-cost-ratios', type=float, nargs='+', default=[0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10], help='C_FP/C_FN ratios')
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists: refusing overwrite')
    seeds = load(a.input/'seed_metrics.csv')
    scenarios = load(a.input/'scenario_metrics.csv')
    expected = {'Global', 'E5-C64', 'E5-C66A'}
    if {r['policy'] for r in seeds} != expected or {r['policy'] for r in scenarios} != expected:
        raise ValueError('Unexpected policy set')
    if {int(r['seed']) for r in seeds} != set(range(6605, 6610)):
        raise ValueError('Unexpected development-check seeds')
    for s in {r['scenario'] for r in seeds}:
        for policy in expected:
            rows = [r for r in seeds if r['scenario']==s and r['policy']==policy]
            if len(rows)!=5: raise ValueError(f'Incomplete scenario/seed/policy: {s} {policy}')
    agg = {}
    for policy in sorted(expected):
        subset = [r for r in seeds if r['policy']==policy]
        agg[policy] = {k:sum(int(r[k]) for r in subset) for k in ('tp','fp','fn')}
    rows=[]
    for ratio in a.fp_cost_ratios:
        if ratio<0: p.error('FP cost ratios must be nonnegative')
        losses = {name: ratio*d['fp']+d['fn'] for name,d in agg.items()}
        winner = min(losses, key=lambda name:(losses[name],name))
        for name,d in agg.items():
            rows.append({'fp_to_fn_cost_ratio':ratio,'policy':name,**d,
                         'weighted_loss':losses[name],
                         'loss_reduction_vs_global':losses['Global']-losses[name],
                         'minimum_loss_policy':winner})
    # Pairwise analytic crossover: A vs B, equal loss when r*(FP_A-FP_B) = FN_B-FN_A.
    cross=[]
    names=sorted(expected)
    for i,a_name in enumerate(names):
        for b_name in names[i+1:]:
            x,y=agg[a_name],agg[b_name]
            denom=x['fp']-y['fp']
            cutoff=(y['fn']-x['fn'])/denom if denom else None
            cross.append({'policy_a':a_name,'policy_b':b_name,
                          'equal_loss_fp_to_fn_ratio':cutoff,
                          'nonnegative_crossover':cutoff is not None and cutoff>=0})
    a.output.mkdir(parents=True)
    save(a.output/'cost_sensitivity.csv',rows)
    save(a.output/'pairwise_crossover.csv',cross)
    (a.output/'summary.json').write_text(json.dumps({
        'experiment':'E5-C68','status':'DEVELOPMENT-ONLY; NO INDEPENDENT TEST',
        'source':str(a.input),'seeds':list(range(6605,6610)),
        'objective':'minimize C_FP * FP + C_FN * FN',
        'assumption':'Each false positive and false negative carries a constant per-edge cost; ratios are illustrative, not empirically calibrated.',
        'aggregates':agg,'crossovers':cross,
        'warning':'Do not select or claim independently validated superiority using E5-C63/E5-C65; prospective policy choice needs operationally justified costs and new held-out testing.'
    },indent=2).encode('utf-8').decode('utf-8'),encoding='utf-8')
    print('E5-C68 | COST-SENSITIVE DEVELOPMENT ANALYSIS')
    for name,d in agg.items(): print(f'{name}: TP={d["tp"]} FP={d["fp"]} FN={d["fn"]}')
    for ratio in a.fp_cost_ratios:
        match=next(r for r in rows if r['fp_to_fn_cost_ratio']==ratio)
        print(f'C_FP/C_FN={ratio:g}: minimum-loss policy={match["minimum_loss_policy"]}')
    for c in cross: print(f'Crossover {c["policy_a"]} vs {c["policy_b"]}: {c["equal_loss_fp_to_fn_ratio"]}')
    print('Saved:',a.output.resolve())
    print('Development only; do not tune against previous independent tests.')

if __name__=='__main__': main()
