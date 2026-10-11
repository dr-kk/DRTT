"""E5-C83: final, read-only consolidation of E5-C79 through E5-C82.

Run: cd D:\\DRTT\\E5_ClosedLoop
     python .\\e5c83_final_consolidation.py

No training, synthetic generation, tuning or new test seeds. Reads existing CSVs,
validates key numerical identities, and writes a frozen reporting package.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / 'results' / 'causal'
OUT = BASE / 'e5c83_final_consolidation'
INPUTS = {
    'C79': BASE / 'e5c79_structure_aware_alert_selection' / 'test_summary.csv',
    'C80': BASE / 'e5c80_component_ablation' / 'test_summary.csv',
    'C81': BASE / 'e5c81_fair_baseline_comparison' / 'test_summary.csv',
    'C82': BASE / 'e5c82_paired_statistical_analysis' / 'paired_comparisons.csv',
    'C82_CI': BASE / 'e5c82_paired_statistical_analysis' / 'method_ci95.csv',
}
EXPECTED = {'C79': (7904, 7908), 'C80': (8001, 8005), 'C81': (8101, 8105)}
SEED_FILES = {
    'C79': BASE / 'e5c79_structure_aware_alert_selection' / 'test_seed_metrics.csv',
    'C80': BASE / 'e5c80_component_ablation' / 'test_seed_metrics.csv',
    'C81': BASE / 'e5c81_fair_baseline_comparison' / 'test_seed_metrics.csv',
}

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty CSV: ' + str(path))
    return rows

def number(row, key):
    return float(row[key])

def get(rows, **conditions):
    matches = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in conditions.items())]
    if len(matches) != 1:
        raise ValueError('Expected one row for %s, got %d' % (conditions, len(matches)))
    return matches[0]

def check_f1(row, source):
    if not {'tp', 'predicted', 'truth', 'pooled_f1'}.issubset(row):
        return
    tp, pred, truth = (number(row, k) for k in ('tp', 'predicted', 'truth'))
    if not (0 <= tp <= min(pred, truth)):
        raise ValueError('Inconsistent TP/predicted/truth: ' + source)
    expected = 2 * tp / (pred + truth) if pred + truth else 0
    if abs(expected - number(row, 'pooled_f1')) > 1e-9:
        raise ValueError('F1 mismatch: ' + source)
    if 'fp' in row and abs(number(row, 'fp') - (pred - tp)) > 1e-9:
        raise ValueError('FP mismatch: ' + source)
    if 'fn' in row and abs(number(row, 'fn') - (truth - tp)) > 1e-9:
        raise ValueError('FN mismatch: ' + source)

def md_table(columns, rows):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines.extend('| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows)
    return '\n'.join(lines)

def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite existing output: ' + str(OUT))
    missing = [str(p) for p in list(INPUTS.values()) + list(SEED_FILES.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing experiment outputs:\n' + '\n'.join(missing))
    data = {name: read_csv(path) for name, path in INPUTS.items()}
    for key, path in SEED_FILES.items():
        rows = read_csv(path)
        actual = sorted({int(r['seed']) for r in rows})
        lo, hi = EXPECTED[key]
        if actual != list(range(lo, hi + 1)):
            raise ValueError('%s expected seeds %d..%d; got %s' % (key, lo, hi, actual))
    for key in ('C79', 'C80', 'C81'):
        for row in data[key]:
            check_f1(row, key)
            if int(row['seeds']) != 5 or int(row['truth']) != 3000 or int(row['predicted']) != 2000:
                raise ValueError('%s violates expected five-seed / fixed-budget summary' % key)

    c79 = data['C79']
    c80 = data['C80']
    c81 = data['C81']
    c82 = data['C82']
    c82_ci = data['C82_CI']
    for variant in ('original', 'matched_signal'):
        a = get(c79, variant=variant, policy='ScoreOnly')
        b = get(c79, variant=variant, policy='TopologyStrong')
        if number(a, 'alerts') != number(b, 'alerts'):
            raise ValueError('C79 budgets differ')
    for variant in ('original', 'matched_signal'):
        for method in ('Frozen', 'HardNegative', 'TemporalHost', 'SeededRandom'):
            get(c81, variant=variant, method=method)
            get(c82_ci, variant=variant, method=method)
    for row in c82:
        if abs(number(row, 'delta_pooled_f1') - (number(row, 'f1_a') - number(row, 'f1_b'))) > 1e-9:
            raise ValueError('C82 comparison arithmetic mismatch')

    out_lines = [
        '# E5-C83 — Final Consolidated Experimental Report',
        '',
        '**Scope:** E5-C79 through E5-C82 only. No new experiment, model fitting, or test-time selection.',
        '**Data:** Five independent synthetic generator seeds per experiment; different seed groups between C79/C80/C81.',
        '**Budgets:** 1,800 alerts and 400 predicted edges per seed in the reported tests.',
        '',
        '## E5-C79: Structure-aware alert selection',
        '',
        md_table(['Telemetry', 'ScoreOnly F1', 'TopologyStrong F1', 'ScoreOnly edge coverage', 'TopologyStrong edge coverage'], [
            [v, '%.4f' % number(get(c79, variant=v, policy='ScoreOnly'), 'pooled_f1'),
             '%.4f' % number(get(c79, variant=v, policy='TopologyStrong'), 'pooled_f1'),
             '%.4f' % number(get(c79, variant=v, policy='ScoreOnly'), 'alert_edge_recall'),
             '%.4f' % number(get(c79, variant=v, policy='TopologyStrong'), 'alert_edge_recall')]
            for v in ('original', 'matched_signal')]),
        '',
        '**Interpretation:** The validation-selected structure-aware policy does not improve held-out F1 or true-edge coverage over score-only selection.',
        '',
        '## E5-C80: Inference-time feature-masking sensitivity',
        '',
        md_table(['Masked component', 'OriginalOnly / original F1', 'OriginalOnly / matched F1'], [
            [a, '%.4f' % number(get(c80, detector='OriginalOnly', variant='original', ablation=a), 'pooled_f1'),
             '%.4f' % number(get(c80, detector='OriginalOnly', variant='matched_signal', ablation=a), 'pooled_f1')]
            for a in ('none', 'temporal', 'topology', 'signal', 'all')]),
        '',
        '**Interpretation:** The frozen model is highly sensitive to topology masking. These are inference-time zero-masking tests, not independently retrained ablations; correlated and derived features may remain.',
        '',
        '## E5-C81: Equal-budget heuristic baseline comparison',
        '',
        md_table(['Method', 'Original F1', 'Matched-signal F1'], [
            [m, '%.4f' % number(get(c81, variant='original', method=m), 'pooled_f1'),
             '%.4f' % number(get(c81, variant='matched_signal', method=m), 'pooled_f1')]
            for m in ('Frozen', 'HardNegative', 'RecentParent', 'HostHandoff', 'TemporalHost', 'SeededRandom')]),
        '',
        '**Interpretation:** Learned models outperform these tested heuristic baselines numerically. These baselines are not published state-of-the-art systems.',
        '',
        '## E5-C82: Paired inference',
        '',
        md_table(['Telemetry', 'Comparison', 'F1 delta', 'Paired bootstrap 95% CI', 'Exact p', 'Holm p'], [
            [r['variant'], r['method_a'] + ' vs ' + r['method_b'],
             '%+.4f' % number(r, 'delta_pooled_f1'),
             '[%+.4f, %+.4f]' % (number(r, 'paired_bootstrap_ci95_low'), number(r, 'paired_bootstrap_ci95_high')),
             '%.4f' % number(r, 'exact_signflip_p_two_sided'),
             '%.4f' % number(r, 'holm_adjusted_p')]
            for r in c82]),
        '',
        '**Inference:** No comparison reaches Holm-adjusted p < 0.05. Bootstrap intervals with only five seed clusters are exploratory; a percentile bootstrap interval excluding zero does not override the non-significant exact test.',
        '',
        '## Limitations and claim boundaries',
        '',
        '1. All results are synthetic; external dataset and deployment generalization are not established.',
        '2. Matched-signal perturbations use benign labels to construct the evaluation distribution; this is an evaluation oracle, not an operational procedure.',
        '3. Alert selection strongly limits true-edge coverage; topology-aware reranking cannot recover omitted edges.',
        '4. Original and matched-signal scores cannot be described as robustly equivalent.',
        '5. Five paired seeds limit inferential resolution; the smallest two-sided exact sign-flip p with five nonzero pairs is 0.0625.',
        '6. C79, C80 and C81 use different independent test seeds. Compare methods within an experiment, not across experiments as if paired.',
        '7. Inference-time masking is not a retrained feature-removal ablation.',
        '8. Heuristic comparisons do not substantiate superiority over published SOTA methods.',
        '',
        '## Suggested manuscript-safe conclusion',
        '',
        'Under equal alert and edge budgets on independent synthetic seeds, the learned reconstruction models achieved higher F1 than the tested heuristic baselines, while hard-negative retraining offered only a small and statistically inconclusive gain over the frozen model. Inference-time masking indicates strong sensitivity to topology-related features. However, structure-aware alert selection did not improve held-out reconstruction, and matched-signal perturbations caused severe performance degradation. These findings identify alert coverage and distribution shift as unresolved limitations rather than solved capabilities.',
        '',
    ]
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'final_report.md').write_text('\n'.join(out_lines), encoding='utf-8')
    # Portable CSV of all final rows, with source experiment included.
    allrows = []
    for k in ('C79', 'C80', 'C81', 'C82'):
        for row in data[k]:
            allrows.append({'experiment': k, **row})
    fields = list(dict.fromkeys(field for row in allrows for field in row))
    with (OUT / 'all_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(allrows)
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in {**INPUTS, **{k+'_seeds':v for k,v in SEED_FILES.items()}}.items()}
    audit = {'experiment': 'E5-C83', 'type': 'read-only final consolidation',
             'source_files': {k: str(v.relative_to(ROOT)) for k,v in INPUTS.items()},
             'source_sha256': hashes, 'test_seed_ranges': EXPECTED,
             'data_integrity': 'summary arithmetic, five-seed identity, fixed truth/prediction totals, and C82 paired delta checked',
             'no_training': True, 'no_new_test_seeds': True,
             'cautions': ['Synthetic only', 'Different seed groups across experiments', 'No Holm-significant differences',
                          'Five seeds insufficient for strong inferential claims', 'Heuristics are not published SOTA']}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        methods = ['Frozen', 'HardNegative', 'RecentParent', 'HostHandoff', 'TemporalHost', 'SeededRandom']
        original = [number(get(c81, variant='original', method=m), 'pooled_f1') for m in methods]
        matched = [number(get(c81, variant='matched_signal', method=m), 'pooled_f1') for m in methods]
        import numpy as np
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(x - .2, original, .4, label='Original telemetry')
        ax.bar(x + .2, matched, .4, label='Matched signal')
        ax.set_xticks(x, methods, rotation=25, ha='right')
        ax.set_ylabel('Pooled causal-edge F1')
        ax.set_title('E5-C81: Fair fixed-budget baseline comparison')
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / 'figure_baselines.png', dpi=220)
        fig.savefig(OUT / 'figure_baselines.pdf')
        plt.close(fig)
        masks = ['none', 'temporal', 'topology', 'signal', 'all']
        vals = [number(get(c80, detector='OriginalOnly', variant='original', ablation=a), 'pooled_f1') for a in masks]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(masks, vals)
        ax.set_ylabel('Pooled causal-edge F1')
        ax.set_title('E5-C80: Inference-time masking (original telemetry)')
        fig.tight_layout()
        fig.savefig(OUT / 'figure_ablations.png', dpi=220)
        fig.savefig(OUT / 'figure_ablations.pdf')
        plt.close(fig)
    except ImportError:
        print('Matplotlib/numpy unavailable: CSV, Markdown and audit generated; figures skipped.')
    print('\nE5-C83 FINAL CONSOLIDATION COMPLETE')
    print('Source checks: PASS')
    print('No new training or experiment: CONFIRMED')
    print('All Holm-adjusted comparisons significant at 0.05:', any(number(r, 'holm_adjusted_p') < .05 for r in c82))
    print('Saved:', OUT)
    for p in sorted(OUT.iterdir()):
        print(' -', p.name)

if __name__ == '__main__':
    main()
