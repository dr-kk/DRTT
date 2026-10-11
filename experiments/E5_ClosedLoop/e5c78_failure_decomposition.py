"""E5-C78: independent synthetic causal reconstruction failure decomposition.
Run from D:\\DRTT\\E5_ClosedLoop. Does not overwrite previous experiments.
Detectors retrained ONLY on E5-C76 train seeds (7601..7610); no fitting on 7801..7805.
Top-k candidate rankings reproduce E5-C58 compatible features and tie-breaking.
All metrics descriptive, no test-based model selection. Matched perturbation is label-informed.
"""
import csv, json, random, sys, hashlib
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76
OUT=ROOT/'results/causal/e5c78_failure_decomposition'
SEEDS=list(range(7801,7806)); ALERTS=[500,900,1300,1800,2400]; EDGES=[200,400,600]
W=90

def write(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def rank_candidates(events,chosen,ranker):
    """Return child -> ordered parent IDs, using exactly the E5-C58 candidate rule."""
    selected=sorted((e for e in events if e['event_id'] in chosen),key=lambda e:(int(e['timestamp']),e['event_id']))
    active=deque(); output={}
    for child in selected:
        t=int(child['timestamp'])
        while active and t-int(active[0]['timestamp'])>W:active.popleft()
        pairs=[]
        for parent in reversed(active):
            gap=t-int(parent['timestamp'])
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            pairs.append((parent,gap))
        pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
        if pairs:
            x=[]
            for i,(parent,gap) in enumerate(pairs):
                x.append([gap/90.,float(parent['source_host']==child['source_host']),float(parent['destination_host']==child['destination_host']),float(parent['destination_host']==child['source_host']),float(parent['source_host']==child['destination_host']),float(parent['sensor']==child['sensor']),float(parent['event_type']==child['event_type']),float(parent['signal']),float(child['signal']),len(pairs)/50.,i/50.])
            x=np.asarray(x,dtype=float)
            z=x if len(ranker['features'])==11 else x[:,:-1]
            scores=np.asarray(ranker['model'].decision_function(z),dtype=float).reshape(-1)
            order=sorted(range(len(pairs)),key=lambda i:(-float(scores[i]),pairs[i][1],pairs[i][0]['event_id']))
            output[child['event_id']]=[pairs[i][0]['event_id'] for i in order]
        active.append(child)
    return output

def summarise(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in ('detector','variant','ranker','alerts','edge_budget'))].append(r)
    ans=[]
    for key,group in grouped.items():
        truth=sum(r['truth'] for r in group); malicious=sum(r['malicious_events'] for r in group)
        alert_edges=sum(r['alert_reachable_edges'] for r in group)
        candidate=sum(r['candidate_reachable_edges'] for r in group)
        t1=sum(r['top1_correct'] for r in group);t3=sum(r['top3_correct'] for r in group)
        tp=sum(r['tp'] for r in group);pred=sum(r['predicted'] for r in group)
        ans.append(dict(zip(('detector','variant','ranker','alerts','edge_budget'),key),seeds=len(group),truth=truth,
          malicious_events=malicious,malicious_alerts=sum(r['malicious_alerts'] for r in group),
          event_recall=sum(r['malicious_alerts'] for r in group)/malicious if malicious else 0,
          alert_reachable_edges=alert_edges,alert_edge_recall=alert_edges/truth if truth else 0,
          candidate_reachable_edges=candidate,candidate_recall_given_alerts=candidate/alert_edges if alert_edges else 0,
          candidate_recall_global=candidate/truth if truth else 0,
          top1_correct=t1,top1_given_candidates=t1/candidate if candidate else 0,
          top3_correct=t3,top3_given_candidates=t3/candidate if candidate else 0,
          predicted=pred,tp=tp,fp=pred-tp,fn=truth-tp,precision=tp/pred if pred else 0,
          recall=tp/truth if truth else 0,pooled_f1=e75.f1(tp,pred,truth)))
    return ans

def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite existing results: {OUT}')
    paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
           'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
           'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
           'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
           'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}
    for k,p in paths.items():
        if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
    old=e75.load(paths['old']);frozen=e75.load(paths['frozen']);hard=e75.load(paths['hard'])
    gate=e75.load(paths['gate']);new_gate=e75.load(paths['new_gate'])
    if old['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1] or hard['features']!=base.FEATURES[:-1]:raise RuntimeError('Ranker feature schema mismatch')
    if gate['features']!=c58.GATE_FEATURES or new_gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Gate schema mismatch')
    if frozen.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['frozen']):raise RuntimeError('Frozen lineage mismatch')
    detectors=e76.fit_models();rankers={'Frozen':frozen,'HardNegative':hard};gates={'Frozen':gate,'HardNegative':new_gate}
    rows=[]
    for seed in SEEDS:
        events,gt=gen.generate(seed,100,2400,10000)
        truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
        labels={r['event_id']:int(r['malicious']) for r in gt}
        benign=[r['signal'] for r in events if labels[r['event_id']]==0]
        rng=random.Random(170000+seed)
        matched=[dict(r,signal=rng.choice(benign)) for r in events]
        for variant,ev in [('original',events),('matched_signal',matched)]:
            ids=[r['event_id'] for r in ev]; feats=c38.extract(ev,W)
            X=np.asarray([feats[i] for i in ids],dtype=float)
            for dname,detector in detectors.items():
                p=detector.predict_proba(X)[:,1]
                sorted_ids=[eid for eid,_ in sorted(zip(ids,p),key=lambda z:(-float(z[1]),z[0]))]
                for alerts in ALERTS:
                    chosen=set(sorted_ids[:alerts]);mal=sum(labels[i] for i in chosen)
                    reachable={(pa,ch) for pa,ch in truth if pa in chosen and ch in chosen}
                    for rname,ranker in rankers.items():
                        ranked_parents=rank_candidates(ev,chosen,ranker)
                        candidate=sum(pa in ranked_parents.get(ch,()) for pa,ch in reachable)
                        top1=sum(bool(ranked_parents.get(ch)) and ranked_parents[ch][0]==pa for pa,ch in reachable)
                        top3=sum(pa in ranked_parents.get(ch,())[:3] for pa,ch in reachable)
                        rr=c58.top_rows(ev,chosen,ranker,W)
                        # Verify that the independent rank calculation matches the gate's top-ranked parent.
                        if len(rr)!=len(ranked_parents) or any(r['parent']!=ranked_parents[r['child']][0] for r in rr):
                            raise RuntimeError(f'Candidate ranking mismatch seed={seed} detector={dname} variant={variant} ranker={rname} alerts={alerts}')
                        if rr:
                            scores=gates[rname]['model'].predict_proba(np.asarray([r['features'] for r in rr],dtype=float))[:,1]
                            ordered=sorted(range(len(rr)),key=lambda i:(-float(scores[i]),rr[i]['child'],rr[i]['parent']))
                            ranked_edges=[(rr[i]['parent'],rr[i]['child']) for i in ordered]
                        else:ranked_edges=[]
                        for budget in EDGES:
                            pred=set(ranked_edges[:budget]);tp=len(pred&truth)
                            rows.append(dict(seed=seed,detector=dname,variant=variant,ranker=rname,alerts=alerts,
                              edge_budget=budget,truth=len(truth),malicious_events=sum(labels.values()),malicious_alerts=mal,
                              alert_reachable_edges=len(reachable),candidate_reachable_edges=candidate,
                              top1_correct=top1,top3_correct=top3,candidate_children=len(rr),predicted=len(pred),tp=tp))
        print(f'test seed={seed} complete',flush=True)
    summary=summarise(rows)
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'test_seed_metrics.csv',rows);write(OUT/'test_summary.csv',summary)
    # 400-edge compact view, useful for inspection; not selected based on test outcomes.
    write(OUT/'fixed400_decomposition.csv',[r for r in summary if r['edge_budget']==400])
    (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C78',test_seeds=SEEDS,
      detector_training_seeds=e76.TRAIN,alert_grid=ALERTS,edge_budget_grid=EDGES,
      input_sha256={k:sha(v) for k,v in paths.items()},
      candidate_rule='E5-C58: selected parents preceding child by 1..90 timestamps with one of three host links',
      denominators={'event_recall':'selected malicious events / all malicious events',
        'alert_edge_recall':'both endpoints alerted / all truth edges',
        'candidate_recall_given_alerts':'true parent in candidate set / both endpoints alerted truth edges',
        'top1_given_candidates':'true parent ranked first / true parent in candidate set',
        'top3_given_candidates':'true parent ranked in first 3 / true parent in candidate set',
        'pooled_f1':'2*TP/(predicted + truth)'},
      limitations=['Synthetic-only','Matched-signal perturbation uses benign ground-truth labels to form counterfactual distribution',
      'Top-k candidate diagnostics use ranker decision function, not gate probabilities',
      'Frozen gate calibration differs between rerankers','All budgets evaluated on test: diagnostic grid, not model selection',
      'No confidence intervals','Candidate search depends on alerts for both endpoints']),indent=2),encoding='utf-8')
    print('\nE5-C78 FIXED 400-EDGE DECOMPOSITION (1800 alerts)')
    for r in summary:
        if r['alerts']==1800 and r['edge_budget']==400:print(r)
    print('Saved:',OUT)
if __name__=='__main__':main()
