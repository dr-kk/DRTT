"""E5-C51: diagnostic decomposition of frozen E5-C50 on seeds 5000-5004.

No model training, selection, or threshold tuning. Ground truth is read only
for diagnostic labels, after alert selection and model predictions.
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


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def selected_candidate_sets(events, chosen, window):
    """Mirror E5-C45/E5-C38 host-continuity candidate eligibility.

    IMPORTANT: the detector selects context-ranked alerts, not signal >= 0.65.
    Calling E5-C15 candidates() on the entire telemetry would silently use
    the wrong alert set. This function uses the frozen context alert IDs.
    """
    ordered = sorted((e for e in events if e['event_id'] in chosen),
                     key=lambda e: (int(e['timestamp']), e['event_id']))
    result = {}
    for j, child in enumerate(ordered):
        ct = int(child['timestamp'])
        entries = []
        for parent in reversed(ordered[:j]):
            gap = ct - int(parent['timestamp'])
            if gap > window:
                break
            if gap <= 0:
                continue
            if not (parent['destination_host'] == child['source_host'] or
                    parent['source_host'] == child['source_host'] or
                    parent['destination_host'] == child['destination_host']):
                continue
            entries.append(parent['event_id'])
        if entries:
            result[child['event_id']] = set(entries)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--joint', type=Path, default=Path('results/causal/e5c49/joint_model.pkl'))
    ap.add_argument('--ranker', type=Path, default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--gate', type=Path, default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--train', type=Path, default=Path('data/causal/e5c15_train'))
    ap.add_argument('--scripts', type=Path, default=Path('src/causal'))
    ap.add_argument('--reference', type=Path, default=Path('results/causal/e5c50/per_seed.csv'))
    ap.add_argument('--output', type=Path, default=Path('results/causal/e5c51'))
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Output directory must be empty; previous results preserved')

    sys.path.insert(0, str(args.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen

    with args.joint.open('rb') as f:
        joint = pickle.load(f)
    with args.ranker.open('rb') as f:
        ranker = pickle.load(f)
    with args.gate.open('rb') as f:
        gate = pickle.load(f)
    if joint['features'] != c41.FEATURES + c38.GATE_FEATURES or len(joint['features']) != 29:
        raise ValueError('Joint feature order mismatch')
    if ranker['features'] != base.FEATURES or gate['features'] != c38.GATE_FEATURES:
        raise ValueError('Ranker/gate feature mismatch')
    if joint['sha256']['ranker'] != sha(args.ranker) or joint['sha256']['gate'] != sha(args.gate):
        raise ValueError('Frozen model provenance mismatch')
    if gate['e5c15_sha256'] != sha(args.ranker):
        raise ValueError('Gate ranker provenance mismatch')
    if joint['sha256']['train_telemetry'] != sha(args.train / 'telemetry.csv'):
        raise ValueError('Training data provenance mismatch')
    if float(ranker['threshold']) != .65 or int(ranker['window']) != 90 or int(gate['window']) != 90:
        raise ValueError('Frozen candidate-generation configuration mismatch')

    reference = {(int(r['seed']), r['pipeline']): r for r in read_csv(args.reference)}
    train = c38.read(args.train / 'telemetry.csv')
    labels = {r['event_id']: int(r['malicious']) for r in c38.read(args.train / 'ground_truth.csv')}
    features = c38.extract(train, 90)
    ids = [r['event_id'] for r in train]
    detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None, random_state=0))
    detector.fit(np.asarray([features[i] for i in ids]), np.asarray([labels[i] for i in ids]))

    per_seed = []
    truth_diagnostics = []
    for seed in range(5000, 5005):
        events, truthrows = gen.generate(seed, 100, 2400, 10000)
        ids = [r['event_id'] for r in events]
        x = c38.extract(events, 90)
        probs = detector.predict_proba(np.asarray([x[i] for i in ids]))[:, 1]
        n = sum(float(r['signal']) >= .65 for r in events)
        chosen = {i for i, _ in sorted(zip(ids, probs), key=lambda z: (-float(z[1]), z[0]))[:n]}
        records = c45.candidates_with_features(events, chosen, ranker, 90)
        expected, audit = c38.top_predictions(events, chosen, ranker, gate, base, 90)
        if len(records) != audit['candidate_children']:
            raise AssertionError(f'Seed {seed}: candidate count mismatch')
        if not records:
            raise AssertionError(f'Seed {seed}: unexpected empty candidate set')
        G = np.asarray([r['gate_features'] for r in records], dtype=float)
        X = np.asarray([r['valid_features'] + r['gate_features'] for r in records], dtype=float)
        if X.shape[1] != 29:
            raise AssertionError('Expected 29 joint features')
        gp = gate['model'].predict_proba(G)[:, 1]
        jp = joint['model'].predict_proba(X)[:, 1]
        baseline = {(r['parent'], r['child']) for r, p in zip(records, gp) if p >= gate['cutoff']}
        joint_edges = {(r['parent'], r['child']) for r, p in zip(records, jp) if p >= .5}
        if baseline != set(expected):
            raise AssertionError(f'Seed {seed}: E5-C38 baseline mismatch')

        truth = {(r['predecessor_event_id'], r['event_id']) for r in truthrows if r['predecessor_event_id']}
        for name, edges in [('context_frozen_gate', baseline), ('context_joint_0_5', joint_edges)]:
            m = c45.evaluate(edges, truth, len(events))
            ref = reference.get((seed, name))
            if ref is None or int(ref['tp']) != m['tp'] or int(ref['fp']) != m['fp']:
                raise AssertionError(f'Seed {seed}: E5-C50 {name} TP/FP mismatch')

        # Recover the *full* eligible candidate sets, rather than inferring
        # availability from the top-1 predictions in E5-C50's CSV.
        candidate_sets = selected_candidate_sets(events, chosen, ranker['window'])
        selected_parent = {r['child']: r['parent'] for r in records}
        if len(selected_parent) != len(records):
            raise AssertionError(f'Seed {seed}: duplicate child among top-1 records')
        if set(selected_parent) != set(candidate_sets):
            raise AssertionError(f'Seed {seed}: candidate-set children differ from top-1 records')
        eligible_parents = candidate_sets

        stages = {'truth_edges': len(truth), 'child_alerted': 0, 'both_alerted': 0,
                  'true_parent_candidate': 0, 'top1_correct': 0,
                  'gate_correct_accepted': 0, 'joint_correct_accepted': 0}
        for parent, child in sorted(truth):
            child_alerted = child in chosen
            both_alerted = child_alerted and parent in chosen
            candidate_available = both_alerted and parent in eligible_parents.get(child, set())
            top1_correct = candidate_available and selected_parent.get(child) == parent
            gate_accepted = (parent, child) in baseline
            joint_accepted = (parent, child) in joint_edges
            if child_alerted: stages['child_alerted'] += 1
            if both_alerted: stages['both_alerted'] += 1
            if candidate_available: stages['true_parent_candidate'] += 1
            if top1_correct: stages['top1_correct'] += 1
            if gate_accepted: stages['gate_correct_accepted'] += 1
            if joint_accepted: stages['joint_correct_accepted'] += 1
            if gate_accepted and not top1_correct or joint_accepted and not top1_correct:
                raise AssertionError('Accepted true edge not top1-correct')
            truth_diagnostics.append({'seed': seed, 'parent': parent, 'child': child,
                                      'child_alerted': int(child_alerted), 'both_alerted': int(both_alerted),
                                      'true_parent_candidate': int(candidate_available),
                                      'top1_correct': int(top1_correct),
                                      'gate_correct_accepted': int(gate_accepted),
                                      'joint_correct_accepted': int(joint_accepted)})
        if stages['gate_correct_accepted'] != int(reference[(seed, 'context_frozen_gate')]['tp']):
            raise AssertionError('Gate correct acceptance mismatch')
        if stages['joint_correct_accepted'] != int(reference[(seed, 'context_joint_0_5')]['tp']):
            raise AssertionError('Joint correct acceptance mismatch')
        if not (stages['gate_correct_accepted'] <= stages['top1_correct'] <= stages['true_parent_candidate'] <= stages['both_alerted'] <= stages['child_alerted'] <= stages['truth_edges']):
            raise AssertionError('Stage ordering invalid')
        per_seed.append({'seed': seed, 'alerts': n, 'candidate_children': len(records), **stages,
                         'lost_child_detection': stages['truth_edges'] - stages['child_alerted'],
                         'lost_parent_detection': stages['child_alerted'] - stages['both_alerted'],
                         'lost_candidate_generation': stages['both_alerted'] - stages['true_parent_candidate'],
                         'lost_ranking': stages['true_parent_candidate'] - stages['top1_correct'],
                         'lost_gate_rejection': stages['top1_correct'] - stages['gate_correct_accepted'],
                         'lost_joint_rejection': stages['top1_correct'] - stages['joint_correct_accepted']})
        print(f'Seed {seed}: truth={stages["truth_edges"]}, both_alerted={stages["both_alerted"]}, '
              f'candidate={stages["true_parent_candidate"]}, top1={stages["top1_correct"]}, '
              f'gate_TP={stages["gate_correct_accepted"]}, joint_TP={stages["joint_correct_accepted"]}')

    fields = ['truth_edges', 'child_alerted', 'both_alerted', 'true_parent_candidate',
              'top1_correct', 'gate_correct_accepted', 'joint_correct_accepted',
              'lost_child_detection', 'lost_parent_detection', 'lost_candidate_generation',
              'lost_ranking', 'lost_gate_rejection', 'lost_joint_rejection']
    totals = {field: int(sum(row[field] for row in per_seed)) for field in fields}
    report = {'experiment': 'E5-C51', 'scope': 'Post-hoc stage decomposition on previously evaluated synthetic seeds 5000-5004',
              'aggregate': totals, 'per_seed': per_seed,
              'checks': {'reproduces_e5c50_tp_fp': True, 'model_or_threshold_tuning': False},
              'sha256': {'joint': sha(args.joint), 'ranker': sha(args.ranker),
                         'gate': sha(args.gate), 'reference': sha(args.reference)},
              'limitations': ['Same synthetic generator; no operational causal ground truth',
                              'These seeds were already evaluated in E5-C50; descriptive diagnostics only',
                              'No retraining, model selection, or threshold tuning',
                              'Stage counts refer to true predecessor edges, not all candidate children',
                              'Detection stage uses the context detector alert set; no ground-truth inference features']}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_seed.csv', per_seed)
    write_csv(args.output / 'truth_edge_diagnostics.csv', truth_diagnostics)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nE5-C51 | RECONSTRUCTION FAILURE DECOMPOSITION')
    print(json.dumps(totals, indent=2))
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
