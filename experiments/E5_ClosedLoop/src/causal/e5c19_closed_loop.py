#!/usr/bin/env python3
"""E5-C19: minimal stateful, paired attacker-defender simulator (stdlib only).
Exploratory mechanics audit, NOT a validated cyber range or measured incident response time.
"""
import argparse, csv, json, random, statistics, time
from pathlib import Path

ACTIONS = ('monitor', 'block', 'isolate', 'preserve')
POLICIES = ('monitor_only', 'block_only', 'isolate_only', 'preserve_only', 'adaptive')
PATHS = ('web', 'remote', 'identity')


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def policy_action(policy, observation):
    if policy == 'monitor_only':
        return 'monitor'
    if policy == 'block_only':
        return 'block' if observation['alert'] else 'monitor'
    if policy == 'isolate_only':
        return 'isolate' if observation['alert'] else 'monitor'
    if policy == 'preserve_only':
        return 'preserve' if observation['alert'] else 'monitor'
    if policy == 'adaptive':
        if observation['alert'] and observation['alert_count'] >= 2:
            return 'isolate'
        if observation['alert']:
            return 'block'
        if observation['uncertain']:
            return 'preserve'
        return 'monitor'
    raise ValueError(policy)


def run_episode(seed, policy, episode, max_steps, missing_rate, conflict_rate):
    # Exogenous draws shared across policies for matched paired comparisons.
    rng = random.Random(seed * 1000003 + episode * 7919)
    path = PATHS[rng.randrange(len(PATHS))]
    initial_path = path
    blocked = set()
    isolated = False
    progress = 0
    switches = 0
    evidence = 0
    lost_evidence = 0
    alerts = 0
    observed_alerts = 0
    trace = []
    completed = False
    for step in range(max_steps):
        # Attacker reacts to previous defender actions before attempting next step.
        if isolated:
            trace.append(dict(episode=episode, step=step, policy=policy, path=path,
                              observation='none', action='none', event='attacker_stopped_isolation',
                              progress=progress, evidence=evidence, wall_ms=0.0))
            break
        if path in blocked:
            alternatives = [p for p in PATHS if p not in blocked]
            if not alternatives:
                trace.append(dict(episode=episode, step=step, policy=policy, path=path,
                                  observation='none', action='none', event='attacker_stopped_blocked',
                                  progress=progress, evidence=evidence, wall_ms=0.0))
                break
            path = alternatives[(episode + step) % len(alternatives)]
            switches += 1
            event = 'attacker_switched_path'
        else:
            event = 'attacker_continues'
        # Fixed exogenous signal schedule by seed/episode/step; no policy-specific RNG.
        signal_rng = random.Random(seed * 1000003 + episode * 7919 + step * 15427 + 7)
        signal = signal_rng.random()
        missing = signal_rng.random() < missing_rate
        conflicting = signal_rng.random() < conflict_rate
        true_alert = signal < 0.72
        alert = (not true_alert if conflicting else true_alert) if not missing else False
        uncertain = missing or conflicting
        if true_alert:
            alerts += 1
        if alert:
            observed_alerts += 1
        observation = {'alert': alert, 'uncertain': uncertain,
                       'alert_count': observed_alerts, 'signal': None if missing else round(signal, 5)}
        start = time.perf_counter_ns()
        action = policy_action(policy, observation)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        # Action modifies future reachable states; observation contains no path/progress labels.
        if action == 'block':
            blocked.add(path)
        elif action == 'isolate':
            isolated = True
        elif action == 'preserve':
            evidence += 1
        else:
            if not missing:
                lost_evidence += 1
        # Attacker already executed the current attempt; action takes effect next step.
        progress += 1
        if progress >= 5:
            completed = True
        trace.append(dict(episode=episode, step=step, policy=policy, path=path,
                          observation=json.dumps(observation, sort_keys=True), action=action,
                          event=event, progress=progress, evidence=evidence,
                          wall_ms=round(elapsed_ms, 6)))
        if completed:
            break
    result = dict(episode=episode, seed=seed, policy=policy, initial_path=initial_path,
                  completed=int(completed), attacker_steps=progress, path_switches=switches,
                  preserved_evidence=evidence, lost_evidence=lost_evidence,
                  true_alert_steps=alerts, observed_alert_steps=observed_alerts,
                  defender_decision_wall_ms=sum(x['wall_ms'] for x in trace))
    return result, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=2050)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--max-steps', type=int, default=8)
    ap.add_argument('--missing-rate', type=float, default=0.10)
    ap.add_argument('--conflict-rate', type=float, default=0.10)
    args = ap.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        ap.error('episodes and max-steps must be positive')
    if not (0 <= args.missing_rate <= 1 and 0 <= args.conflict_rate <= 1):
        ap.error('missing-rate and conflict-rate must be within [0,1]')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results, traces = [], []
    for episode in range(args.episodes):
        for policy in POLICIES:
            result, trace = run_episode(args.seed, policy, episode, args.max_steps,
                                        args.missing_rate, args.conflict_rate)
            results.append(result)
            traces.extend(trace)
    write_csv(out / 'episodes.csv', results, list(results[0]))
    write_csv(out / 'trace.csv', traces, ['episode','step','policy','path','observation','action',
                                         'event','progress','evidence','wall_ms'])
    summary = {'seed': args.seed, 'episodes_per_policy': args.episodes,
               'policies': {}, 'limitations': [
                   'Stylized synthetic state machine, not real network attack execution',
                   'Wall time measures Python policy decision computation, not incident response time',
                   'Path identity is logged for evaluator but never provided to policy',
                   'Preserved evidence count is not forensic reconstruction accuracy',
                   'Actions are deterministic heuristics, not trained RL policies']}
    for policy in POLICIES:
        group = [r for r in results if r['policy'] == policy]
        summary['policies'][policy] = {
            'completion_rate': statistics.mean(r['completed'] for r in group),
            'mean_attacker_steps': statistics.mean(r['attacker_steps'] for r in group),
            'mean_path_switches': statistics.mean(r['path_switches'] for r in group),
            'mean_preserved_evidence': statistics.mean(r['preserved_evidence'] for r in group),
            'mean_decision_wall_ms_per_episode': statistics.mean(r['defender_decision_wall_ms'] for r in group)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('E5-C19 | INTERACTIVE CAUSAL MECHANICS SMOKE TEST')
    print(json.dumps(summary, indent=2))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
