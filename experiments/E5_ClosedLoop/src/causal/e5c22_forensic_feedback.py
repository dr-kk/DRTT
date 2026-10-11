#!/usr/bin/env python3
"""E5-C22 controlled synthetic forensic-feature ablation; not a network testbed.

Policies have identical score-to-action logic. Forensics variant adds a feature
from verified, causally prior evidence records. Calibration selects thresholds
using separate seeded episodes, then evaluates paired held-out episodes.
"""
import argparse, csv, hashlib, json, math, random, statistics, time
from pathlib import Path

POLICIES=('monitor_only','isolate_only','adaptive_no_forensics','adaptive_with_forensics')
PATHS=('web','remote','identity')

def save_csv(path, rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def scenario(seed,ep,benign_fraction):
    r=random.Random(seed*1000003+ep*7919)
    benign=r.random()<benign_fraction
    return {'benign':benign,'initial_path':PATHS[r.randrange(3)],'seed':seed,'episode':ep}

def observation(seed,ep,step,benign,missing_rate,conflict_rate,tamper_rate):
    # Exogenous randomness is shared by all policies for each episode and step.
    r=random.Random(seed*1000003+ep*7919+step*15427+7)
    score=min(.99,max(.01,r.gauss(.26 if benign else .71,.23)))
    missing=r.random()<missing_rate
    conflict=r.random()<conflict_rate
    tamper=r.random()<tamper_rate
    if conflict:score=1-score
    alert=(score>=.56) and not missing
    # Sensor-observable event categories. No hidden path, label or progress.
    category=('credential' if r.random()<(.16 if benign else .70) else 'routine')
    if missing:category='missing'
    record={'score':None if missing else round(score,5),'alert':alert,
            'category':category,'missing':missing,'conflict':conflict}
    payload=json.dumps(record,sort_keys=True)
    seal=hashlib.sha256(payload.encode()).hexdigest()
    # Integrity check uses independent sealed payload; tamper invalidates record.
    return record,seal,(not missing and not tamper)

def evidence_feature(history):
    """Only verified observed evidence from current and preceding steps.

    A reconstructed *temporal motif* is two consecutive verified credential
    observations. It is not ground-truth causal attack-chain recovery.
    """
    if len(history)<2:return 0.0
    motifs=sum(1 for a,b in zip(history,history[1:])
               if b['step']==a['step']+1 and a['category']=='credential'
               and b['category']=='credential')
    return min(1.0,motifs/2)

def action_for(policy,score,threshold):
    if policy=='monitor_only':return 'monitor'
    if policy=='isolate_only':return 'isolate' if score>=threshold else 'monitor'
    if score>=threshold:return 'isolate'
    if score>=threshold-.23:return 'block'
    if score>=threshold-.40:return 'preserve'
    return 'monitor'

def run_episode(cfg,policy,threshold,weight,args,with_trace=False):
    seed=cfg['seed'];ep=cfg['episode'];benign=cfg['benign']
    path=cfg['initial_path'];blocked=set();history=[];steps=0;switches=0
    cost=0.0;preserved=0;false_isolation=0;completed=0;isolated=0
    trace=[];decision_ms=0.0
    for step in range(args.max_steps):
        if path in blocked:
            options=[p for p in PATHS if p not in blocked]
            if not options:break
            path=options[(ep+step)%len(options)];switches+=1
        obs,seal,valid=observation(seed,ep,step,benign,args.missing_rate,args.conflict_rate,args.tamper_rate)
        if valid:
            history.append({'step':step,'category':obs['category'],'seal':seal})
        base=obs['score'] if obs['score'] is not None else 0.0
        motif=evidence_feature(history)
        # The same scoring and action code is used by both adaptive policies;
        # the ablation toggles the availability of the forensic feature.
        use_forensics=policy=='adaptive_with_forensics'
        decision_score=min(1.0,max(0.0,base+(weight*motif if use_forensics else 0.0)))
        t0=time.perf_counter_ns()
        action=action_for(policy,decision_score,threshold)
        decision_ms+=(time.perf_counter_ns()-t0)/1e6
        cost+={'monitor':0.0,'block':1.0,'preserve':.4,'isolate':5.0}[action]
        if action=='block':blocked.add(path)
        if action=='preserve' and valid:preserved+=1
        if action=='isolate':
            isolated=1
            if benign:false_isolation=1
        steps+=1
        # Isolation takes effect after the current observation/action cycle.
        if not benign and steps>=5:completed=1
        if with_trace:
            trace.append({'seed':seed,'episode':ep,'policy':policy,'step':step,
                'observed_score':obs['score'],'category':obs['category'],
                'integrity_valid':int(valid),'motif_score':motif,
                'decision_score':round(decision_score,5),'action':action,
                'private_benign':int(benign),'private_path':path,
                'private_completed':completed})
        if isolated or completed:break
    # Cost is deliberately illustrative, with availability penalty for false isolation.
    utility_loss=(20.0*completed+10.0*false_isolation+cost)
    row={'seed':seed,'episode':ep,'policy':policy,'benign':int(benign),
        'attack_completed':completed,'false_isolation':false_isolation,
        'steps':steps,'path_switches':switches,'preserved_evidence':preserved,
        'action_cost':round(cost,3),'utility_loss':round(utility_loss,3),
        'decision_wall_ms':round(decision_ms,6)}
    return row,trace

def mean(rows,key):return statistics.mean(r[key] for r in rows) if rows else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--calibration-seed',type=int,default=2053)
    p.add_argument('--test-seed',type=int,default=2054)
    p.add_argument('--calibration-episodes',type=int,default=300)
    p.add_argument('--test-episodes',type=int,default=500)
    p.add_argument('--max-steps',type=int,default=8)
    p.add_argument('--benign-fraction',type=float,default=.5)
    p.add_argument('--missing-rate',type=float,default=.1)
    p.add_argument('--conflict-rate',type=float,default=.1)
    p.add_argument('--tamper-rate',type=float,default=.05)
    a=p.parse_args()
    if a.calibration_seed==a.test_seed:p.error('calibration and test seeds must differ')
    if min(a.calibration_episodes,a.test_episodes,a.max_steps)<1:p.error('counts must be positive')
    if any(not 0<=v<=1 for v in (a.benign_fraction,a.missing_rate,a.conflict_rate,a.tamper_rate)):
        p.error('rates must lie in [0,1]')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cal=[scenario(a.calibration_seed,i,a.benign_fraction) for i in range(a.calibration_episodes)]
    test=[scenario(a.test_seed,i,a.benign_fraction) for i in range(a.test_episodes)]
    # Pre-registered candidate grid; select separately on calibration only.
    grid=[(t,w) for t in (.55,.65,.75,.85) for w in (0.0,.10,.20,.30)]
    selected={};cal_rows=[]
    for pol in POLICIES:
        candidates=[]
        for t,w in grid:
            if pol!='adaptive_with_forensics' and w!=0:continue
            rows=[run_episode(c,pol,t,w,a)[0] for c in cal]
            loss=mean(rows,'utility_loss')
            candidates.append((loss,t,w))
            cal_rows.append({'policy':pol,'threshold':t,'forensic_weight':w,'calibration_loss':round(loss,6)})
        loss,t,w=min(candidates)
        selected[pol]={'threshold':t,'forensic_weight':w,'calibration_loss':loss}
    results=[];traces=[]
    for c in test:
        for pol in POLICIES:
            setting=selected[pol]
            row,trace=run_episode(c,pol,setting['threshold'],setting['forensic_weight'],a,True)
            results.append(row);traces.extend(trace)
    save_csv(out/'calibration_grid.csv',cal_rows)
    save_csv(out/'episodes.csv',results)
    save_csv(out/'trace.csv',traces)
    summary={'design':'paired held-out synthetic test; calibration seed separate from test seed',
             'calibration_seed':a.calibration_seed,'test_seed':a.test_seed,
             'calibration_episodes':len(cal),'test_episodes':len(test),
             'calibration':selected,'policies':{},'paired_difference':{},
             'limitations':['Simulated event scores, not measured network traffic',
             'Forensic feature is a verified temporal motif, not causal attack reconstruction',
             'Ground truth appears only in private evaluation columns and simulator state',
             'Action costs and loss weights are assumed, not measured',
             'No trained RL/PPO and no operational incident-response latency',
             'Single held-out seed; repeat independent seeds for uncertainty estimates']}
    for pol in POLICIES:
        subset=[r for r in results if r['policy']==pol]
        mal=[r for r in subset if not r['benign']];ben=[r for r in subset if r['benign']]
        summary['policies'][pol]={
            'attack_completion_rate':mean(mal,'attack_completed'),
            'false_isolation_rate':mean(ben,'false_isolation'),
            'mean_action_cost':mean(subset,'action_cost'),
            'mean_utility_loss':mean(subset,'utility_loss'),
            'mean_preserved_evidence':mean(subset,'preserved_evidence'),
            'mean_decision_wall_ms':mean(subset,'decision_wall_ms')}
    for key in ('attack_completed','false_isolation','action_cost','utility_loss','preserved_evidence'):
        pairs=[]
        for i in range(len(test)):
            d={r['policy']:r for r in results if r['episode']==i}
            if key=='attack_completed' and d['adaptive_no_forensics']['benign']:continue
            if key=='false_isolation' and not d['adaptive_no_forensics']['benign']:continue
            pairs.append(d['adaptive_with_forensics'][key]-d['adaptive_no_forensics'][key])
        summary['paired_difference'][key]={'mean':statistics.mean(pairs) if pairs else None,
             'sample_sd':statistics.stdev(pairs) if len(pairs)>1 else None,
             'n':len(pairs),'nonzero_pairs':sum(x!=0 for x in pairs)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('E5-C22 | CONTROLLED FORENSIC FEEDBACK EVALUATION')
    print(json.dumps(summary,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
