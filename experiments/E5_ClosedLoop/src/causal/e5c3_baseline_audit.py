"""E5-C3: fair, fixed-threshold comparison of predecessor-link heuristics.

Inference receives only telemetry; labels are loaded after all predictions.
This is an exploratory same-dataset diagnostic, NOT a held-out benchmark.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ALLOWED = {'event_id','campaign_id','timestamp','source_host','destination_host','sensor','event_type','signal'}

def load(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def infer(events, threshold, max_gap, mode, seed=2026):
    if not events or set(events[0]) != ALLOWED:
        raise ValueError('Telemetry schema invalid; labels must never enter inference')
    groups = defaultdict(list)
    for e in events:
        if set(e) != ALLOWED:
            raise ValueError('Unexpected telemetry columns')
        groups[e['campaign_id']].append(e)
    selected_ids, edges = set(), set()
    rng = random.Random(seed)
    for campaign, rows in sorted(groups.items()):
        selected = []
        for e in sorted(rows, key=lambda r: (int(r['timestamp']), r['event_id'])):
            if float(e['signal']) < threshold:
                continue
            selected_ids.add(e['event_id'])
            t = int(e['timestamp'])
            candidates = []
            for prev in selected:
                gap = t - int(prev['timestamp'])
                if gap < 0 or gap > max_gap:
                    continue
                continuity = prev['destination_host'] == e['source_host']
                same_host = (prev['source_host'] == e['source_host'] or
                             prev['destination_host'] == e['destination_host'])
                if mode == 'graph' and not (continuity or same_host):
                    continue
                score = ((0.6 if continuity else 0.3) +
                         (0.25 if same_host else 0) +
                         0.15 * (1 - gap / max_gap))
                candidates.append((score, -gap, prev['event_id']))
            if candidates and mode != 'none':
                if mode == 'graph':
                    predecessor = max(candidates)[2]
                elif mode == 'nearest':
                    predecessor = max(candidates, key=lambda c: (c[1], c[2]))[2]
                elif mode == 'random':
                    predecessor = rng.choice(candidates)[2]
                else:
                    raise ValueError(mode)
                edges.add((predecessor, e['event_id']))
            selected.append(e)
    return selected_ids, edges

def metrics(selected, edges, truth):
    by_id = {r['event_id']: int(r['malicious']) for r in truth}
    if len(by_id) != len(truth) or not selected <= set(by_id):
        raise ValueError('Invalid ground-truth event IDs')
    tp = sum(by_id[i] for i in selected)
    fp = len(selected) - tp
    fn = sum(by_id.values()) - tp
    tn = len(by_id) - tp - fp - fn
    true_edges = {(r['predecessor_event_id'], r['event_id'])
                  for r in truth if r['predecessor_event_id']}
    etp = len(edges & true_edges)
    return {
        'event_f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0,
        'event_fpr': fp/(fp+tn) if (fp+tn) else 0,
        'predicted_edges': len(edges), 'true_edges': len(true_edges),
        'edge_tp': etp,
        'edge_precision': etp/len(edges) if edges else 0,
        'edge_recall': etp/len(true_edges) if true_edges else 0,
        'edge_f1': 2*etp/(len(edges)+len(true_edges)) if (len(edges)+len(true_edges)) else 0,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.65)
    p.add_argument('--max-gap', type=int, default=90)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1 or args.max_gap <= 0:
        p.error('Invalid threshold/max-gap')
    events = load(args.telemetry)
    # All four inference runs finish before ground truth is opened.
    results = {mode: infer(events, args.threshold, args.max_gap, mode)
               for mode in ('none', 'nearest', 'random', 'graph')}
    truth = load(args.truth)
    report = {mode: metrics(*result, truth) for mode, result in results.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'e5c3_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('E5-C3 | MATCHED HEURISTIC BASELINES (exploratory, not held-out)')
    print(f"{'method':<12} {'edges':>7} {'correct':>8} {'edge P':>9} {'edge R':>9} {'edge F1':>9}")
    for mode, m in report.items():
        print(f"{mode:<12} {m['predicted_edges']:>7} {m['edge_tp']:>8} "
              f"{m['edge_precision']:>9.4f} {m['edge_recall']:>9.4f} {m['edge_f1']:>9.4f}")
    print('Event F1 and FPR are identical across modes (fixed detector).')
    print('Saved:', args.output/'e5c3_comparison.json')

if __name__ == '__main__':
    main()
