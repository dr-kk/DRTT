#!/usr/bin/env python3
"""E5-C24: frozen synthetic chain-feature ablation with shuffled negative control.
Not real causal reconstruction, real network evidence, or RL.
"""
import argparse,csv,json,random,statistics,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import importlib.util

MODES=('no_forensics','temporal_motif','chain_reconstruction','shuffled_chain')

def module_at(path):
    spec=importlib.util.spec_from_file_location('e5c22_source',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def save(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def reconstruct(history):
    """Infer directed consecutive-event links from verified observable categories only.
    The candidate links are heuristic temporal links, NOT proven causal edges.
    """
    edges=[]
    for a,b in zip(history,history[1:]):
        if b['step']-a['step']!=1:continue
        if a['category']=='credential' and b['category']=='credential':
            edges.append((a['step'],b['step']))
    # Nonlinear confidence in a coherent chain; no private labels or path.
    if not edges:return 0.0,edges
    max_run=1;run=1
    for old,new in zip(edges,edges[1:]):
        run=run+1 if old[1]==new[0] else 1
        max_run=max(max_run,run)
    return min(1.0,max_run/2),edges

def episode(mod,cfg,mode,args,shuffled_categories):
    seed=cfg['seed'];ep=cfg['episode'];benign=cfg['benign']
    path=cfg['initial_path'];blocked=set();history=[];switches=0;cost=0.0
    preserved=0;false_isolation=0;completed=0;isolated=0;steps=0;trace=[]
    for step in range(args.max_steps):
        if path in blocked:
            options=[p for p in mod.PATHS if p not in blocked]
            if not options:break
            path=options[(ep+step)%len(options)];switches+=1
        obs,seal,valid=mod.observation(seed,ep,step,benign,args.missing_rate,args.conflict_rate,args.tamper_rate)
        if valid:
            # Only current and preceding valid observations are accessible.
            history.append({'step':step,'category':obs['category'],'seal':seal})
        base=obs['score'] if obs['score'] is not None else 0.0
        motif=mod.evidence_feature(history)
        chain,edges=reconstruct(history)
        # Negative control: replace category sequence with a separately seeded
        # episode's observable category sequence, preserving local valid steps.
        shuffled_history=[dict(h,category=shuffled_categories[h['step']]) for h in history]
        shuffled,shuffled_edges=reconstruct(shuffled_history)
        feature={'no_forensics':0.0,'temporal_motif':motif,
                 'chain_reconstruction':chain,'shuffled_chain':shuffled}[mode]
        decision_score=max(0.0,min(1.0,base+args.forensic_weight*feature))
        action=mod.action_for('adaptive_with_forensics',decision_score,args.threshold)
        cost+={'monitor':0.,'block':1.,'preserve':.4,'isolate':5.}[action]
        if action=='block':blocked.add(path)
        if action=='preserve' and valid:preserved+=1
        if action=='isolate':
            isolated=1
            if benign:false_isolation=1
        steps+=1
        if not benign and steps>=5:completed=1
        trace.append(dict(seed=seed,episode=ep,mode=mode,step=step,
            score=obs['score'],verified=int(valid),category=obs['category'],
            reconstructed_edge_count=len(edges),feature=feature,
            decision_score=round(decision_score,5),action=action,
            private_benign=int(benign),private_path=path))
        if isolated or completed:break
    utility=20*completed+10*false_isolation+cost
    row=dict(seed=seed,episode=ep,mode=mode,benign=int(benign),
        attack_completed=completed,false_isolation=false_isolation,
        action_cost=round(cost,4),utility_loss=round(utility,4),
        preserved_evidence=preserved,steps=steps,path_switches=switches)
    return row,trace

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='src/causal/e5c22_forensic_feedback.py')
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[2070,2071,2072,2073,2074])
    p.add_argument('--episodes',type=int,default=500)
    p.add_argument('--max-steps',type=int,default=8)
    p.add_argument('--benign-fraction',type=float,default=.5)
    p.add_argument('--missing-rate',type=float,default=.1)
    p.add_argument('--conflict-rate',type=float,default=.1)
    p.add_argument('--tamper-rate',type=float,default=.05)
    p.add_argument('--threshold',type=float,default=.85)
    p.add_argument('--forensic-weight',type=float,default=.30)
    a=p.parse_args()
    if len(set(a.seeds))<2 or a.episodes<2 or a.max_steps<2:p.error('Need >=2 unique seeds and >=2 episodes/steps')
    if set(a.seeds)&{2053,2054,2060,2061,2062,2063,2064}:p.error('Use fresh seeds, not calibration or prior test seeds')
    if any(not 0<=x<=1 for x in (a.benign_fraction,a.missing_rate,a.conflict_rate,a.tamper_rate)):
        p.error('Rates must lie in [0,1]')
    if not Path(a.source).is_file():p.error('E5-C22 source not found')
    mod=module_at(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];traces=[];per_seed=[]
    for seed in a.seeds:
        configs=[mod.scenario(seed,ep,a.benign_fraction) for ep in range(a.episodes)]
        # Independent donor category streams for negative control. Not truth.
        donor={}
        for cfg in configs:
            ep=cfg['episode'];donor_ep=(ep+max(1,a.episodes//2))%a.episodes
            donor_cfg=configs[donor_ep]
            donor[ep]=[mod.observation(seed,donor_ep,step,donor_cfg['benign'],
                a.missing_rate,a.conflict_rate,a.tamper_rate)[0]['category']
                for step in range(a.max_steps)]
        for cfg in configs:
            for mode in MODES:
                r,t=episode(mod,cfg,mode,a,donor[cfg['episode']]);rows.append(r);traces.extend(t)
        for mode in MODES:
            sub=[r for r in rows if r['seed']==seed and r['mode']==mode]
            mal=[r for r in sub if not r['benign']];ben=[r for r in sub if r['benign']]
            if not mal or not ben:raise RuntimeError('Seed contains no benign or malicious episodes')
            per_seed.append(dict(seed=seed,mode=mode,
                malicious_n=len(mal),benign_n=len(ben),
                attack_completion_rate=statistics.mean(r['attack_completed'] for r in mal),
                false_isolation_rate=statistics.mean(r['false_isolation'] for r in ben),
                mean_utility_loss=statistics.mean(r['utility_loss'] for r in sub),
                mean_action_cost=statistics.mean(r['action_cost'] for r in sub),
                mean_preserved_evidence=statistics.mean(r['preserved_evidence'] for r in sub)))
    save(out/'episodes.csv',rows);save(out/'trace.csv',traces);save(out/'per_seed.csv',per_seed)
    keys=('attack_completion_rate','false_isolation_rate','mean_utility_loss','mean_action_cost','mean_preserved_evidence')
    comparisons={}
    for mode in MODES[1:]:
        comparisons[mode]={}
        for key in keys:
            diffs=[]
            for seed in a.seeds:
                d={r['mode']:r for r in per_seed if r['seed']==seed}
                diffs.append(d[mode][key]-d['no_forensics'][key])
            comparisons[mode][key]=dict(mean_difference=statistics.mean(diffs),
                per_seed=dict(zip(map(str,a.seeds),diffs)),
                sd_across_seeds=statistics.stdev(diffs))
    # Critical identification check: reconstructed feature and temporal motif
    # may coincide, in which case E5-C24 adds NO new reconstruction evidence.
    motif_lookup={(t['seed'],t['episode'],t['step']):t['feature']
                  for t in traces if t['mode']=='temporal_motif'}
    chain_traces=[t for t in traces if t['mode']=='chain_reconstruction']
    same=sum(abs(t['feature']-motif_lookup.get((t['seed'],t['episode'],t['step']),-100))<1e-12
             for t in chain_traces)
    chain_count=len(chain_traces)
    summary=dict(design='Synthetic paired frozen-feature ablation; no re-calibration',
        seeds=a.seeds,episodes_per_seed=a.episodes,threshold=a.threshold,
        forensic_weight=a.forensic_weight,comparisons_vs_no_forensics=comparisons,
        chain_feature_equals_motif_on_matching_observed_steps=same,
        chain_feature_comparisons=chain_count,
        limitations=['Heuristic temporal edges are not independently validated causal chains',
            'Chain feature may coincide with temporal motif: check equality count',
            'Donor-shuffled category negative control changes feature distribution',
            'Action costs synthetic; no real IRT or PPO; no causal ground-truth edge scoring'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('E5-C24 | CHAIN FEATURE AND NEGATIVE CONTROL')
    print(json.dumps(summary,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
