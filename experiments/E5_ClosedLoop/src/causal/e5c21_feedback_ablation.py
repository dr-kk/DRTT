#!/usr/bin/env python3
"""E5-C21: paired synthetic forensic-feedback intervention audit.
No real network execution, RL, or operational IRT measurement.
"""
import argparse,csv,json,random,statistics,hashlib,time
from pathlib import Path

POLICIES=('monitor_only','isolate_only','adaptive_no_forensics','adaptive_with_forensics')
PATHS=('web','remote','identity')

def save(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def episode(seed,ep,policy,steps,missing_rate,conflict_rate,tamper_rate):
    r=random.Random(seed*1000003+ep*7919)
    initial=PATHS[r.randrange(3)]
    path=initial; blocked=set();isolated=False;progress=0;switches=0
    alerts=0;verified=[]; preserved=0;cost=0;traces=[];completed=False
    # Every policy gets identical exogenous observations for a given episode/step.
    # Fingerprint is a *sensor-observable category*, not the private attacker path.
    for step in range(steps):
        if isolated:break
        if path in blocked:
            options=[p for p in PATHS if p not in blocked]
            if not options:break
            path=options[(ep+step)%len(options)];switches+=1
        x=random.Random(seed*1000003+ep*7919+step*15427+7)
        signal=x.random();missing=x.random()<missing_rate
        conflict=x.random()<conflict_rate;tampered=x.random()<tamper_rate
        actual_alert=signal<.72
        alert=(not actual_alert if conflict else actual_alert) if not missing else False
        fingerprint='high' if signal<.36 else ('mid' if signal<.72 else 'low')
        # Synthetic artifact seal; defender sees only valid records from past steps.
        payload=f'{ep}|{step}|{fingerprint}|{int(alert)}'
        digest=hashlib.sha256(payload.encode()).hexdigest()
        valid=not missing and not tampered
        observation={'alert':alert,'uncertain':missing or conflict,'signal':None if missing else round(signal,5)}
        # Verification is performed before decision; no private path, progress or labels.
        if valid:
            verified.append((step,fingerprint,alert,digest))
        if alert:alerts+=1
        recent_verified=sum(int(v[2]) for v in verified[-3:])
        t0=time.perf_counter_ns()
        if policy=='monitor_only':action='monitor'
        elif policy=='isolate_only':action='isolate' if alert else 'monitor'
        elif policy=='adaptive_no_forensics':
            action='isolate' if alert and alerts>=2 else ('block' if alert else ('preserve' if observation['uncertain'] else 'monitor'))
        else:
            # Same policy skeleton, but escalation is based on integrity-verified,
            # recent alert evidence rather than an unverified running alert count.
            action='isolate' if alert and recent_verified>=2 else ('block' if alert else ('preserve' if observation['uncertain'] else 'monitor'))
        decision_ms=(time.perf_counter_ns()-t0)/1e6
        # Explicit opportunity costs; illustrative utility assumptions only.
        action_cost={'monitor':0,'block':1,'preserve':.5,'isolate':5}[action]
        cost+=action_cost
        if action=='block':blocked.add(path)
        if action=='isolate':isolated=True
        if action=='preserve' and valid:preserved+=1
        progress+=1
        completed=progress>=5
        traces.append(dict(seed=seed,episode=ep,policy=policy,step=step,
            observation=json.dumps(observation,sort_keys=True),verified_recent_alerts=recent_verified,
            verified_evidence_count=len(verified),action=action,action_cost=action_cost,
            decision_wall_ms=round(decision_ms,6),private_path=path,private_progress=progress,
            private_true_alert=int(actual_alert),completed=int(completed)))
        if completed:break
    return dict(seed=seed,episode=ep,policy=policy,initial_path=initial,completed=int(completed),
        attacker_steps=progress,path_switches=switches,preserved_evidence=preserved,
        verified_artifacts=len(verified),action_cost=round(cost,4),
        decision_wall_ms=round(sum(t['decision_wall_ms'] for t in traces),6)),traces

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=2052)
    a.add_argument('--episodes',type=int,default=100);a.add_argument('--max-steps',type=int,default=8)
    a.add_argument('--missing-rate',type=float,default=.1);a.add_argument('--conflict-rate',type=float,default=.1)
    a.add_argument('--tamper-rate',type=float,default=.05)
    x=a.parse_args()
    if x.episodes<1 or x.max_steps<1:a.error('episodes and max-steps must be positive')
    if any(not 0<=v<=1 for v in (x.missing_rate,x.conflict_rate,x.tamper_rate)):a.error('rates must be in [0,1]')
    out=Path(x.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];trace=[]
    for ep in range(x.episodes):
        for policy in POLICIES:
            r,t=episode(x.seed,ep,policy,x.max_steps,x.missing_rate,x.conflict_rate,x.tamper_rate)
            rows.append(r);trace.extend(t)
    save(out/'episodes.csv',rows);save(out/'trace.csv',trace)
    summary={'design':'paired synthetic intervention, identical exogenous seed/episode/step draws',
       'seed':x.seed,'episodes_per_policy':x.episodes,
       'policies':{},'limitations':[
       'Simplified state machine; not real attacker or network telemetry',
       'Forensic feedback uses verified recent alerts, not reconstructed attack chains',
       'SHA-256 digest is internally generated, not an external custody guarantee',
       'Action costs are arbitrary illustrative units, not measured service disruption',
       'Decision wall time is Python policy computation, not incident response time',
       'Policies are hand-coded heuristics, not reinforcement learning']}
    for pol in POLICIES:
        subset=[r for r in rows if r['policy']==pol]
        summary['policies'][pol]={k:statistics.mean(r[k] for r in subset) for k in
          ('completed','attacker_steps','path_switches','preserved_evidence','verified_artifacts','action_cost','decision_wall_ms')}
    paired={}
    for ep in range(x.episodes):
        d={r['policy']:r for r in rows if r['episode']==ep}
        f=d['adaptive_with_forensics'];n=d['adaptive_no_forensics']
        for key in ('completed','attacker_steps','action_cost','preserved_evidence'):
            paired.setdefault(key,[]).append(f[key]-n[key])
    summary['paired_forensics_minus_no_forensics']={k:{
        'mean_difference':statistics.mean(v),
        'sample_sd':statistics.stdev(v) if len(v)>1 else None,
        'nonzero_pairs':sum(bool(z) for z in v)} for k,v in paired.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C21 | FORENSIC FEEDBACK ABLATION');print(json.dumps(summary,indent=2))
    print('Saved:',out.resolve())
if __name__=='__main__':main()
