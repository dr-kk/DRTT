"""E5-C6: campaign-independent, telemetry-only heuristic reconstruction.
Ground truth is read only after all predictions have been generated.
Exploratory fixed settings; no hyperparameter optimization or causal proof.
"""
import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

OBSERVED = {'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}

def read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def infer(rows, threshold, window, seed):
    if not rows or set(rows[0]) != OBSERVED:
        raise ValueError('Telemetry schema must contain exactly the seven observable fields')
    ids = [r['event_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate event IDs')
    stream = sorted(rows, key=lambda r: (int(r['timestamp']), r['event_id']))
    rng = random.Random(seed)
    recent = deque()
    edges = {'nearest': set(), 'random': set(), 'graph': set()}
    positives = set()
    for e in stream:
        t = int(e['timestamp'])
        while recent and t - int(recent[0]['timestamp']) > window:
            recent.popleft()
        if float(e['signal']) < threshold:
            continue
        positives.add(e['event_id'])
        # Require strictly earlier timestamp: do not fabricate within-tick order.
        candidates = [p for p in recent if int(p['timestamp']) < t]
        if candidates:
            nearest = max(candidates, key=lambda p: (int(p['timestamp']), p['event_id']))
            edges['nearest'].add((nearest['event_id'], e['event_id']))
            random_parent = rng.choice(candidates)
            edges['random'].add((random_parent['event_id'], e['event_id']))
            scored = []
            for p in candidates:
                continuity = p['destination_host'] == e['source_host']
                same_src = p['source_host'] == e['source_host']
                same_dst = p['destination_host'] == e['destination_host']
                if not (continuity or same_src or same_dst):
                    continue
                gap = t - int(p['timestamp'])
                score = (0.6 if continuity else 0.0) + (0.25 if same_src or same_dst else 0.0) + 0.15*(1-gap/window)
                scored.append((score, -gap, p['event_id']))
            if scored:
                parent = max(scored)[2]
                edges['graph'].add((parent, e['event_id']))
        recent.append(e)
    return positives, edges

def evaluate(positives, edges, truth):
    truth_ids = {r['event_id'] for r in truth}
    if len(truth_ids) != len(truth) or not positives <= truth_ids:
        raise ValueError('Ground truth IDs invalid')
    malicious = {r['event_id'] for r in truth if int(r['malicious']) == 1}
    tp = len(positives & malicious)
    fp = len(positives - malicious)
    fn = len(malicious - positives)
    tn = len(truth_ids) - tp - fp - fn
    true_edges = {(r['predecessor_event_id'], r['event_id']) for r in truth if r['predecessor_event_id']}
    result = {'event': {'TP':tp,'FP':fp,'TN':tn,'FN':fn,
                       'precision':tp/(tp+fp) if tp+fp else 0,
                       'recall':tp/(tp+fn) if tp+fn else 0,
                       'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                       'fpr':fp/(fp+tn) if fp+tn else 0},
              'true_edges':len(true_edges), 'methods':{}}
    for name, pred in edges.items():
        hit = len(pred & true_edges)
        result['methods'][name] = {'predicted_edges':len(pred),'TP':hit,
            'precision':hit/len(pred) if pred else 0,
            'recall':hit/len(true_edges) if true_edges else 0,
            'f1':2*hit/(len(pred)+len(true_edges)) if pred or true_edges else 0}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--truth', type=Path, help='Evaluation only; optional')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.65)
    p.add_argument('--window', type=int, default=90)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1 or args.window < 1:
        p.error('threshold must be in [0,1], window must be positive')
    events = read(args.telemetry)
    positives, edges = infer(events, args.threshold, args.window, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pred in edges.items():
        with (args.output / f'{name}_edges.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['predecessor_event_id','event_id'])
            w.writerows(sorted(pred))
    with (args.output / 'event_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['event_id','predicted_malicious'])
        w.writerows((r['event_id'],int(r['event_id'] in positives)) for r in events)
    print('E5-C6 | CAMPAIGN-INDEPENDENT INFERENCE')
    print('Events:',len(events),'Predicted positives:',len(positives))
    for name, pred in edges.items():
        print(name, 'predicted edges:',len(pred))
    print('Inference used telemetry only; no campaign ID or hidden labels.')
    if args.truth:
        result = evaluate(positives, edges, read(args.truth))
        with (args.output/'evaluation.json').open('w', encoding='utf-8') as f:
            json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
    print('Saved:',args.output)

if __name__ == '__main__':
    main()
