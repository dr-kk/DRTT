"""E5-C54: frozen-ranker hard-negative parent audit on E5-C53 development seeds.

Descriptive synthetic-data analysis. Ground truth used for post-inference labels only.
No retraining, threshold optimization, or evaluation on frozen test seeds.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = range(5200, 5205)
BUDGETS = (1.0, 1.25, 1.5, 2.0, 3.0)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('No data for ' + str(path))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relation(parent, child):
    return {
        'gap': int(child['timestamp']) - int(parent['timestamp']),
        'parent_signal': float(parent['signal']),
        'same_source': int(parent['source_host'] == child['source_host']),
        'same_destination': int(parent['destination_host'] == child['destination_host']),
        'dst_to_src': int(parent['destination_host'] == child['source_host']),
        'src_to_dst': int(parent['source_host'] == child['destination_host']),
        'same_sensor': int(parent['sensor'] == child['sensor']),
        'same_type': int(parent['event_type'] == child['event_type']),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker', type=Path, default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--gate', type=Path, default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--reference', type=Path, default=Path('results/causal/e5c53/per_seed.csv'))
    ap.add_argument('--train', type=Path, default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts', type=Path, default=Path('src/causal'))
    ap.add_argument('--output', type=Path, default=Path('results/causal/e5c54'))
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Output directory not empty; existing results will not be overwritten')

    sys.path.insert(0, str(args.scripts.resolve()))
    import e5c38_end_to_end as c38
    import e5c53_candidate_density_ranking_audit as c53
    import e5c5_overlapping_generator as gen

    with args.ranker.open('rb') as f:
        ranker = pickle.load(f)
    with args.gate.open('rb') as f:
        gate = pickle.load(f)
    if gate['e5c15_sha256'] != sha256(args.ranker):
        raise ValueError('Frozen gate/ranker SHA256 mismatch')
    if float(ranker['threshold']) != .65 or int(ranker['window']) != 90:
        raise ValueError('Unexpected frozen ranker configuration')

    with args.reference.open(newline='', encoding='utf-8-sig') as f:
        reference = {(int(r['seed']), float(r['budget_multiplier'])): r for r in csv.DictReader(f)}
    if len(reference) != 25:
        raise ValueError('Expected 25 unique E5-C53 reference rows')

    train = c38.read(args.train / 'telemetry.csv')
    labels = {r['event_id']: int(r['malicious']) for r in c38.read(args.train / 'ground_truth.csv')}
    train_features = c38.extract(train, 90)
    train_ids = [r['event_id'] for r in train]
    detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None, random_state=0))
    detector.fit(np.asarray([train_features[i] for i in train_ids]), np.asarray([labels[i] for i in train_ids]))

    cases = []
    per_seed = []
    for seed in SEEDS:
        events, truthrows = gen.generate(seed, 100, 2400, 10000)
        event_map = {r['event_id']: r for r in events}
        true_parent = {r['event_id']: r['predecessor_event_id'] for r in truthrows if r['predecessor_event_id']}
        ids = [r['event_id'] for r in events]
        features = c38.extract(events, 90)
        probabilities = detector.predict_proba(np.asarray([features[i] for i in ids]))[:, 1]
        alert_order = [i for i, _ in sorted(zip(ids, probabilities), key=lambda z: (-float(z[1]), z[0]))]
        base_budget = sum(float(r['signal']) >= .65 for r in events)
        for budget_multiplier in BUDGETS:
            n_alerts = min(len(events), int(round(base_budget * budget_multiplier)))
            chosen = set(alert_order[:n_alerts])
            candidate_ranks = c53.candidate_ranks(events, chosen, ranker, 90)
            eligible = 0
            correct = 0
            hard_negative = 0
            for child_id, candidates in candidate_ranks.items():
                parent_id = true_parent.get(child_id)
                if parent_id not in candidates:
                    continue
                eligible += 1
                true_rank = candidates.index(parent_id) + 1
                if true_rank == 1:
                    correct += 1
                    continue
                hard_negative += 1
                wrong_id = candidates[0]
                child = event_map[child_id]
                true_rel = relation(event_map[parent_id], child)
                wrong_rel = relation(event_map[wrong_id], child)
                row = {
                    'seed': seed, 'budget_multiplier': budget_multiplier,
                    'child': child_id, 'true_parent': parent_id, 'incorrect_top1': wrong_id,
                    'candidate_count': len(candidates), 'true_parent_rank': true_rank,
                    'wrong_parent_is_malicious': int(wrong_id in {r['event_id'] for r in truthrows if int(r['malicious']) == 1}),
                }
                for name in true_rel:
                    row['true_' + name] = true_rel[name]
                    row['wrong_' + name] = wrong_rel[name]
                    row['wrong_minus_true_' + name] = wrong_rel[name] - true_rel[name]
                row['wrong_temporally_closer'] = int(wrong_rel['gap'] < true_rel['gap'])
                row['wrong_higher_signal'] = int(wrong_rel['parent_signal'] > true_rel['parent_signal'])
                cases.append(row)
            ref = reference[(seed, budget_multiplier)]
            if eligible != int(ref['eligible_truth_edges']) or correct != int(ref['top1_correct']):
                raise AssertionError(f'E5-C53 mismatch seed={seed} budget={budget_multiplier}: '
                                     f'eligible {eligible}/{ref["eligible_truth_edges"]}, '
                                     f'top1 {correct}/{ref["top1_correct"]}')
            per_seed.append({'seed': seed, 'budget_multiplier': budget_multiplier,
                             'eligible_truth_edges': eligible, 'top1_correct': correct,
                             'hard_negative_cases': hard_negative,
                             'top1_error_rate': hard_negative / eligible if eligible else 0})
            print(f'Seed {seed} x{budget_multiplier:g}: eligible={eligible} '
                  f'top1_correct={correct} hard_negatives={hard_negative}')

    summary = []
    for budget in BUDGETS:
        subset = [r for r in cases if r['budget_multiplier'] == budget]
        seed_rows = [r for r in per_seed if r['budget_multiplier'] == budget]
        eligible = sum(r['eligible_truth_edges'] for r in seed_rows)
        errors = len(subset)
        row = {'budget_multiplier': budget, 'eligible_truth_edges': eligible,
               'top1_correct': eligible - errors, 'hard_negative_cases': errors,
               'top1_error_rate': errors / eligible if eligible else 0}
        for field in ('candidate_count', 'true_parent_rank', 'wrong_temporally_closer',
                      'wrong_higher_signal', 'wrong_parent_is_malicious'):
            row['mean_' + field] = float(np.mean([r[field] for r in subset])) if subset else None
        for field in ('gap', 'parent_signal', 'same_source', 'same_destination',
                      'dst_to_src', 'src_to_dst', 'same_sensor', 'same_type'):
            for prefix in ('true_', 'wrong_', 'wrong_minus_true_'):
                row['mean_' + prefix + field] = (float(np.mean([r[prefix + field] for r in subset]))
                                                      if subset else None)
        summary.append(row)

    report = {
        'experiment': 'E5-C54',
        'description': 'Descriptive comparison of incorrect top-1 parent with true parent, only when true parent is in candidate set',
        'seeds': list(SEEDS), 'budgets': list(BUDGETS),
        'summary': summary,
        'sha256': {'ranker': sha256(args.ranker), 'gate': sha256(args.gate),
                   'reference': sha256(args.reference), 'train_telemetry': sha256(args.train / 'telemetry.csv')},
        'limitations': [
            'Synthetic development seeds from the same generator; not independent validation',
            'Only failures where true parent is eligible are analyzed; this is a conditional sample',
            'Ground truth is used only after frozen model inference to identify hard negatives',
            'Descriptive comparisons are not causal attribution and may be confounded',
            'No retraining, gate changes, or threshold optimization',
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'hard_negative_cases.csv', cases)
    write_csv(args.output / 'per_seed.csv', per_seed)
    write_csv(args.output / 'budget_summary.csv', summary)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nE5-C54 | HARD-NEGATIVE PARENT AUDIT')
    for r in summary:
        print(f'x{r["budget_multiplier"]:g}: eligible={r["eligible_truth_edges"]} '
              f'errors={r["hard_negative_cases"]} error_rate={r["top1_error_rate"]:.3f} '
              f'wrong_closer={r["mean_wrong_temporally_closer"]:.3f} '
              f'wrong_higher_signal={r["mean_wrong_higher_signal"]:.3f}')
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
