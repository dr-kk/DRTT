"""E5-C33: descriptive paired analysis of an already frozen E5-C32 report.
No retraining, re-evaluation, or test-driven model selection.
"""
import argparse, csv, hashlib, json, math
from pathlib import Path
from statistics import mean, stdev

# Two-sided Student-t 0.975 critical value for n=5 (df=4).
T_CRIT_DF4 = 2.7764451051977987
METRICS = ('precision', 'recall', 'f1', 'false_edges_per_1000_events')


def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_summary(values):
    n = len(values)
    avg = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    half = T_CRIT_DF4 * sd / math.sqrt(n)
    return {'n': n, 'mean': avg, 'sample_sd': sd,
            'descriptive_95pct_t_interval': [avg - half, avg + half],
            'all_positive': all(v > 0 for v in values),
            'all_negative': all(v < 0 for v in values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='E5-C32 summary.json')
    p.add_argument('--output', type=Path, required=True, help='New output folder')
    args = p.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8'))
    rows = source.get('per_dataset')
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError('Expected exactly five per_dataset records in E5-C32 summary.json')
    ids = [r['dataset'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate dataset names')
    for row in rows:
        for method in ('baseline', 'gate'):
            entry = row[method]
            if entry['predicted'] != entry['tp'] + entry['fp']:
                raise ValueError(f'Invalid confusion counts in {row["dataset"]} {method}')
            if entry['fn'] != row['truth_edges'] - entry['tp']:
                raise ValueError(f'Invalid false-negative count in {row["dataset"]} {method}')
    stats = {}
    csv_rows = []
    for metric in METRICS:
        diffs = [float(r['gate'][metric]) - float(r['baseline'][metric]) for r in rows]
        stats[metric] = paired_summary(diffs)
        for row, delta in zip(rows, diffs):
            csv_rows.append({'dataset': row['dataset'], 'metric': metric,
                             'baseline': row['baseline'][metric], 'gate': row['gate'][metric],
                             'gate_minus_baseline': delta})
    fp_base = sum(r['baseline']['fp'] for r in rows)
    fp_gate = sum(r['gate']['fp'] for r in rows)
    report = {
        'design': 'Post-hoc descriptive paired seed analysis; no model selection',
        'source': str(args.input), 'source_sha256': filehash(args.input),
        'dataset_ids': ids, 'datasets': len(rows),
        'pooled_baseline': source['baseline'], 'pooled_gate': source['gate'],
        'paired_gate_minus_baseline': stats,
        'false_positive_reduction_fraction': (fp_base - fp_gate) / fp_base if fp_base else None,
        'notes': [
            'Student-t intervals use five seed-level paired differences and df=4.',
            'Intervals are descriptive; five synthetic realizations are not real-world validation.',
            'The generator and test seed provenance are as documented in E5-C32.',
            'This is a post-hoc analysis of previously viewed test results; do not use to select or tune models.',
            'Precision and F1 intervals are in fractions; multiply by 100 for percentage points.',
        ]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out_json = args.output / 'summary.json'
    out_csv = args.output / 'paired_per_seed.csv'
    out_md = args.output / 'publication_table.md'
    for path in (out_json, out_csv, out_md):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing report: {path}')
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    with out_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'metric', 'baseline', 'gate', 'gate_minus_baseline'])
        writer.writeheader()
        writer.writerows(csv_rows)
    def pct(v): return f'{100*v:.2f}%'
    table = [
        '| Metric | Frozen E5-C17 | E5-C31 gate | Paired mean change (95% descriptive CI) |',
        '|---|---:|---:|---:|',
    ]
    for metric, title in [('precision','Precision'), ('recall','Recall'), ('f1','Edge F1'),
                          ('false_edges_per_1000_events','False edges / 1,000 events')]:
        st = stats[metric]
        m = st['mean']; lo, hi = st['descriptive_95pct_t_interval']
        if metric == 'false_edges_per_1000_events':
            fmt = lambda v: f'{v:.2f}'
        else:
            fmt = pct
        table.append(f'| {title} | {fmt(source["baseline"][metric])} | {fmt(source["gate"][metric])} | '
                     f'{fmt(m)} [{fmt(lo)}, {fmt(hi)}] |')
    table.extend(['', 'Five paired synthetic seeds; intervals are descriptive, not operational guarantees.',
                  'No training, calibration, or test-driven model selection was performed by E5-C33.'])
    out_md.write_text('\n'.join(table) + '\n', encoding='utf-8')
    print('E5-C33 | PAIRED DESCRIPTIVE STATISTICS')
    for metric, st in stats.items():
        print(f'{metric}: mean difference={st["mean"]:.6f}, '
              f'95% descriptive interval={st["descriptive_95pct_t_interval"]}')
    print(f'False-positive reduction: {100*report["false_positive_reduction_fraction"]:.2f}%')
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
