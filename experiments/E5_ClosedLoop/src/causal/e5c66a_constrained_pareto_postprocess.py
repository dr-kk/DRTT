"""E5-C66A: Recompute the Pareto frontier restricted to feasible E5-C66 grid points."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dominates(a, b, tol=1e-12):
    f1_a, f1_b = float(a['macro_f1']), float(b['macro_f1'])
    rec_a, rec_b = float(a['macro_recall']), float(b['macro_recall'])
    fp_a, fp_b = int(a['total_fp']), int(b['total_fp'])
    no_worse = f1_a >= f1_b-tol and rec_a >= rec_b-tol and fp_a <= fp_b
    strictly_better = f1_a > f1_b+tol or rec_a > rec_b+tol or fp_a < fp_b
    return no_worse and strictly_better


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='E5-C66 threshold_grid.csv')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError('Input grid is empty')
    required = {'sparse_cutoff', 'dense_cutoff', 'macro_f1', 'macro_recall', 'total_fp', 'feasible'}
    if not required.issubset(rows[0]):
        raise ValueError(f'Missing columns: {sorted(required-set(rows[0]))}')
    feasible = [r for r in rows if int(r['feasible']) == 1]
    frontier = [r for r in feasible if not any(dominates(other, r) for other in feasible if other is not r)]
    frontier.sort(key=lambda r: (int(r['total_fp']), -float(r['macro_f1'])))
    best = max(frontier, key=lambda r: (float(r['macro_f1']), -int(r['total_fp']))) if frontier else None
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output/'feasible_threshold_pairs.csv', feasible)
    write_rows(args.output/'constrained_pareto_frontier.csv', frontier)
    summary = {
        'experiment': 'E5-C66A',
        'method': 'Pareto dominance evaluated only within feasible E5-C66 grid points',
        'source_grid': str(args.input),
        'total_grid_points': len(rows),
        'feasible_grid_points': len(feasible),
        'constrained_pareto_points': len(frontier),
        'best_feasible_by_macro_f1': best,
        'status': 'DEVELOPMENT POSTPROCESSING ONLY; NOT AN INDEPENDENT VALIDATION',
    }
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('E5-C66A | CONSTRAINED PARETO POSTPROCESSING')
    print(f'Grid={len(rows)} Feasible={len(feasible)} Constrained-Pareto={len(frontier)}')
    if best:
        print('Best feasible pair: sparse={sparse_cutoff} dense={dense_cutoff} macro_F1={macro_f1} total_FP={total_fp}'.format(**best))
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
