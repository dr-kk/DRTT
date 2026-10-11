"""E5-C52 | Exploratory alert-budget sensitivity (same-generator synthetic data).

Five development seeds 5200..5204. The detector, reranker, and E5-C31 gate
are frozen. Alert budgets are prespecified, not selected by test performance.
No ground-truth fields are used to select alerts or infer parent links.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BUDGETS = (1.0, 1.25, 1.5, 2.0, 3.0)
SEEDS = tuple(range(5200, 5205))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('No output rows')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def selected_candidate_sets(events, chosen, window):
    """Diagnostic host-continuity eligibility matching E5-C38/C45.

    No signal threshold is reapplied: chosen already contains context alerts.
    """
    ordered = sorted((r for r in events if r['event_id'] in chosen),
                     key=lambda r: (int(r['timestamp']), r['event_id']))
    sets = {}
    for j, child in enumerate(ordered):
        ct = int(child['timestamp'])
        parents = set()
        for parent in reversed(ordered[:j]):
            gap = ct - int(parent['timestamp'])
            if gap > window:
                break
            if gap <= 0:
                continue
            if (parent['destination_host'] == child['source_host'] or
                parent['source_host'] == child['source_host'] or
                parent['destination_host'] == child['destination_host']):
                parents.add(parent['event_id'])
        if parents:
            sets[child['event_id']] = parents
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker', type=Path, default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--gate', type=Path, default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--train', type=Path, default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts', type=Path, default=Path('src/causal'))
    ap.add_argument('--output', type=Path, default=Path('results/causal/e5c52'))
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Output directory must be empty: existing results are preserved')

    sys.path.insert(0, str(args.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen

    with args.ranker.open('rb') as f:
        ranker = pickle.load(f)
    with args.gate.open('rb') as f:
        gate = pickle.load(f)
    if gate['e5c15_sha256'] != sha(args.ranker):
        raise ValueError('Gate-ranker SHA256 mismatch')
    if ranker['features'] != base.FEATURES or gate['features'] != c38.GATE_FEATURES:
        raise ValueError('Frozen feature schema mismatch')
    if (float(ranker['threshold']) != .65 or int(ranker['window']) != 90 or
            int(gate['window']) != 90):
        raise ValueError('Unexpected frozen threshold/window')

    train = c38.read(args.train / 'telemetry.csv')
    labels = {r['event_id']: int(r['malicious']) for r in c38.read(args.train / 'ground_truth.csv')}
    features = c38.extract(train, 90)
    ids = [r['event_id'] for r in train]
    detector = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight=None, random_state=0))
    detector.fit(np.asarray([features[i] for i in ids]),
                 np.asarray([labels[i] for i in ids]))

    rows = []
    for seed in SEEDS:
        events, truthrows = gen.generate(seed, 100, 2400, 10000)
        ids = [r['event_id'] for r in events]
        feat = c38.extract(events, 90)
        probs = detector.predict_proba(np.asarray([feat[i] for i in ids]))[:, 1]
        ranked = [i for i, p in sorted(zip(ids, probs), key=lambda z: (-float(z[1]), z[0]))]
        base_budget = sum(float(r['signal']) >= .65 for r in events)
        truth = {(r['predecessor_event_id'], r['event_id'])
                 for r in truthrows if r['predecessor_event_id']}
        malicious = {r['event_id'] for r in truthrows if int(r['malicious']) == 1}
        for multiplier in BUDGETS:
            budget = min(len(events), int(round(base_budget * multiplier)))
            chosen = set(ranked[:budget])
            records = c45.candidates_with_features(events, chosen, ranker, 90)
            candidate_sets = selected_candidate_sets(events, chosen, 90)
            if len(candidate_sets) != len(records):
                raise AssertionError(f'Seed {seed} budget {multiplier}: candidate count mismatch')
            if {r['child'] for r in records} != set(candidate_sets):
                raise AssertionError('Candidate children mismatch')
            if records:
                gp = gate['model'].predict_proba(
                    np.asarray([r['gate_features'] for r in records], dtype=float))[:, 1]
                accepted = {(r['parent'], r['child']) for r, p in zip(records, gp)
                            if p >= gate['cutoff']}
            else:
                accepted = set()
            if multiplier == 1.0:
                expected, audit = c38.top_predictions(events, chosen, ranker, gate, base, 90)
                if accepted != set(expected) or len(records) != audit['candidate_children']:
                    raise AssertionError(f'Seed {seed}: baseline mismatch with E5-C38')
            child_alerted = sum(child in chosen for parent, child in truth)
            both_alerted = sum(parent in chosen and child in chosen for parent, child in truth)
            parent_candidate = sum(parent in candidate_sets.get(child, set())
                                   for parent, child in truth)
            top1 = {(r['parent'], r['child']) for r in records}
            top1_correct = len(top1 & truth)
            m = c45.evaluate(accepted, truth, len(events))
            row = {
                'seed': seed, 'budget_multiplier': multiplier,
                'base_alert_budget': base_budget, 'alerts': len(chosen),
                'malicious_alerts': len(chosen & malicious),
                'alert_precision': len(chosen & malicious) / len(chosen) if chosen else 0,
                'event_detection_recall': len(chosen & malicious) / len(malicious),
                'truth_edges': len(truth), 'child_alerted': child_alerted,
                'both_alerted': both_alerted, 'true_parent_candidate': parent_candidate,
                'top1_correct': top1_correct, 'candidate_children': len(records),
                'ranking_accuracy_given_candidate': top1_correct / parent_candidate if parent_candidate else 0,
                'correct_edge_acceptance_given_top1': m['tp'] / top1_correct if top1_correct else 0,
                **m
            }
            if not (m['tp'] <= top1_correct <= parent_candidate <= both_alerted <= child_alerted <= len(truth)):
                raise AssertionError('Invalid stage counts')
            rows.append(row)
            print(f'Seed {seed} budget x{multiplier:g}: alerts={budget}, '
                  f'both={both_alerted}, top1={top1_correct}, '
                  f'TP/FP={m["tp"]}/{m["fp"]}, F1={m["f1"]:.4f}')

    summary = []
    for multiplier in BUDGETS:
        group = [r for r in rows if r['budget_multiplier'] == multiplier]
        tp = sum(r['tp'] for r in group)
        fp = sum(r['fp'] for r in group)
        total_truth = sum(r['truth_edges'] for r in group)
        s = {'budget_multiplier': multiplier,
             'alerts_mean': float(np.mean([r['alerts'] for r in group])),
             'event_detection_recall_mean': float(np.mean([r['event_detection_recall'] for r in group])),
             'both_alerted': sum(r['both_alerted'] for r in group),
             'true_parent_candidate': sum(r['true_parent_candidate'] for r in group),
             'top1_correct': sum(r['top1_correct'] for r in group),
             'tp': tp, 'fp': fp,
             'precision_pooled': tp / (tp + fp) if tp + fp else 0,
             'recall_pooled': tp / total_truth if total_truth else 0,
             'f1_pooled': 2 * tp / (2 * tp + fp + total_truth - tp) if total_truth else 0,
             'f1_mean': float(np.mean([r['f1'] for r in group])),
             'false_edges_per_1000_events_mean': float(np.mean([
                 r['false_edges_per_1000_events'] for r in group]))}
        summary.append(s)
    report = {
        'experiment': 'E5-C52',
        'scope': 'Exploratory same-generator synthetic sensitivity, seeds 5200-5204; not independent confirmation',
        'alert_budget_multipliers': BUDGETS, 'seeds': SEEDS,
        'selection_rule': 'Frozen context detector probability top-N; N=round(multiplier * count(signal >= 0.65))',
        'summary': summary,
        'sha256': {'ranker': sha(args.ranker), 'gate': sha(args.gate),
                   'train_telemetry': sha(args.train / 'telemetry.csv')},
        'limitations': [
            'Synthetic generator only; no operational or causal attribution validation',
            'Seeds and budgets are exploratory; do not select and report the best as independently validated',
            'Ground truth used for diagnosis/evaluation only, never detector or reconstruction inference',
            'Higher alert budgets change the candidate population and false-attribution opportunity',
            'Event/edge observations within seeds are dependent; do not treat as independent replicates'
        ]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_seed.csv', rows)
    write_csv(args.output / 'budget_summary.csv', summary)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nE5-C52 | DETECTOR RECALL - RECONSTRUCTION SENSITIVITY')
    for s in summary:
        print(f'x{s["budget_multiplier"]:g}: detection_recall={s["event_detection_recall_mean"]:.3f} '
              f'both={s["both_alerted"]} top1={s["top1_correct"]} '
              f'TP/FP={s["tp"]}/{s["fp"]} F1={s["f1_mean"]:.4f}')
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
