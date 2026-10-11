"""E5-C85: label-informed synthetic host-overlap stress test (evaluation only).
Run from D:\\DRTT\\E5_ClosedLoop. Never overwrites existing results.
Benign host pairs are sampled from attack event host-pair distribution; attack
telemetry and truth edges remain unchanged. This is a controlled diagnostic,
NOT an independently generated or label-free counterfactual.
"""
import csv, json, hashlib, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76

SEEDS=list(range(8501,8511)); ALERTS=1800; EDGES=400; W=90
OUT=ROOT/'results/causal/e5c85_host_overlap_stress'
PATHS={'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
       'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
       'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
       'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}

def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def overlap_variant(events,truth_rows,seed):
    labels={r['event_id']:int(r['malicious']) for r in truth_rows}
    attack_pairs=[(r['source_host'],r['destination_host']) for r in events if labels[r['event_id']]]
    assert attack_pairs and len(labels)==len(events)
    rng=random.Random(985000+seed)
    changed=[]
    for r in events:
        x=dict(r)
        if not labels[r['event_id']]:
            x['source_host'],x['destination_host']=rng.choice(attack_pairs)
        changed.append(x)
    for a,b in zip(events,changed):
        for k in ('event_id','timestamp','sensor','event_type','signal'):
            assert a[k]==b[k]
        if labels[a['event_id']]:assert a==b
    return changed

def reachable(edges,chosen,events):
    byid={r['event_id']:r for r in events}
    selected={(p,c) for p,c in edges if p in chosen and c in chosen}
    compatible=0
    for p,c in selected:
        pa,ch=byid[p],byid[c]
        gap=int(ch['timestamp'])-int(pa['timestamp'])
        if 0<gap<=W and (pa['destination_host']==ch['source_host'] or pa['source_host']==ch['source_host'] or pa['destination_host']==ch['destination_host']):compatible+=1
    return len(selected),compatible

def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite {OUT}')
    for name,p in PATHS.items():
        if not p.is_file():raise FileNotFoundError(f'{name}: {p}')
    models={name:e75.load(p) for name,p in PATHS.items()}
    if models['gate']['features']!=c58.GATE_FEATURES or models['new_gate']['features']!=c58.GATE_FEATURES:
        raise RuntimeError('Gate feature schema mismatch')
    detectors=e76.fit_models(); detector=detectors['OriginalOnly']
    rows=[]; paired=[]
    for seed in SEEDS:
        events,gt=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
        stress=overlap_variant(events,gt,seed)
        assert len(truth)==600
        for variant,ev in [('original',events),('benign_host_overlap',stress)]:
            ids=[r['event_id'] for r in ev]
            features=c38.extract(ev,W)
            scores=detector.predict_proba(np.asarray([features[i] for i in ids],dtype=float))[:,1]
            prob_by_id=dict(zip(ids,map(float,scores)))
            chosen=set(sorted(ids,key=lambda i:(-prob_by_id[i],i))[:ALERTS])
            alert_reach,compatible_reach=reachable(truth,chosen,ev)
            for name,rkey,gkey in [('Frozen','frozen','gate'),('HardNegative','hard','new_gate')]:
                rr=c58.top_rows(ev,chosen,models[rkey],W)
                if rr:
                    gp=models[gkey]['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1]
                    pred=e75.top_budget(rr,gp,EDGES)
                else:pred=set()
                tp=len(pred&truth)
                rows.append(dict(seed=seed,variant=variant,model=name,alerts=len(chosen),truth=len(truth),
                    edge_budget=EDGES,predicted=len(pred),tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,
                    precision=tp/len(pred) if pred else 0.,recall=tp/len(truth),f1=e75.f1(tp,len(pred),len(truth)),
                    alert_reachable=alert_reach,host_compatible_reachable=compatible_reach,
                    candidate_children=len(rr)))
        print(f'seed={seed} original/stress complete',flush=True)
    groups=defaultdict(list)
    for r in rows:groups[(r['variant'],r['model'])].append(r)
    summary=[]
    for (variant,name),rr in groups.items():
        tp=sum(x['tp'] for x in rr);pred=sum(x['predicted'] for x in rr);truth=sum(x['truth'] for x in rr)
        summary.append(dict(variant=variant,model=name,seeds=len(rr),truth=truth,predicted=pred,tp=tp,fp=pred-tp,fn=truth-tp,
            precision=tp/pred if pred else 0.,recall=tp/truth,pooled_f1=e75.f1(tp,pred,truth),
            alert_reachable=sum(x['alert_reachable'] for x in rr),
            host_compatible_reachable=sum(x['host_compatible_reachable'] for x in rr)))
    for name in ('Frozen','HardNegative'):
        for seed in SEEDS:
            a=next(r for r in rows if r['seed']==seed and r['variant']=='original' and r['model']==name)
            b=next(r for r in rows if r['seed']==seed and r['variant']=='benign_host_overlap' and r['model']==name)
            paired.append(dict(seed=seed,model=name,delta_f1=b['f1']-a['f1'],delta_tp=b['tp']-a['tp'],
                               delta_alert_reachable=b['alert_reachable']-a['alert_reachable'],
                               delta_host_compatible_reachable=b['host_compatible_reachable']-a['host_compatible_reachable']))
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'per_seed.csv',rows);write(OUT/'summary.csv',summary);write(OUT/'paired_deltas.csv',paired)
    (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C85',seeds=SEEDS,alerts=ALERTS,edges=EDGES,
        perturbation='Replace benign source/destination host pairs by seeded samples from the attack host-pair marginal; leave all attack events, timestamps, event IDs, sensors, types, signals and truth unchanged',
        provenance='Truth-informed stress construction; truth not passed to detector/ranker/gate inference',
        model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in PATHS.items()},
        limitations=['Synthetic-only; no retraining','Attack host pairs used to construct counterfactual: label-informed evaluation','Not a generator retrained with attack/benign shared-host distributions','Attack host pairs unchanged; benign traffic distribution shifted','Host-compatible reachable edges and selected alerts may change','Descriptive ten-seed paired analysis; no formal significance claim']),indent=2),encoding='utf-8')
    for r in summary:print(r)
    print('Saved:',OUT)
if __name__=='__main__':main()
