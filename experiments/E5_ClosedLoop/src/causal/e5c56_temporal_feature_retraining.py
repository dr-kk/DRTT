"""E5-C56: train-only proximity-rank removal; development ranking audit.
No new-seed evaluation here: model selection must precede independent testing.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS=range(5200,5205)
BUDGETS=(1.,1.25,1.5,2.,3.)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,rows):
    if not rows:raise ValueError('Empty '+str(path))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rank_candidates(events,chosen,original,new_model,window):
    ordered=sorted((e for e in events if e['event_id'] in chosen),key=lambda e:(int(e['timestamp']),e['event_id']))
    result={'original':{},'no_gap_rank':{}}
    for j,child in enumerate(ordered):
        t=int(child['timestamp']);pairs=[]
        for parent in reversed(ordered[:j]):
            gap=t-int(parent['timestamp'])
            if gap>window:break
            if gap<=0:continue
            if not (parent['destination_host']==child['source_host'] or parent['source_host']==child['source_host'] or parent['destination_host']==child['destination_host']):continue
            pairs.append((parent,gap))
        pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
        if not pairs:continue
        vec=[]
        for i,(parent,gap) in enumerate(pairs):
            vec.append([gap/90.,int(parent['source_host']==child['source_host']),int(parent['destination_host']==child['destination_host']),int(parent['destination_host']==child['source_host']),int(parent['source_host']==child['destination_host']),int(parent['sensor']==child['sensor']),int(parent['event_type']==child['event_type']),float(parent['signal']),float(child['signal']),len(pairs)/50.,i/50.])
        x=np.asarray(vec,dtype=float)
        for name,model,features in [('original',original['model'],x),('no_gap_rank',new_model,x[:,:-1])]:
            scores=np.asarray(model.decision_function(features)).reshape(-1)
            order=sorted(range(len(pairs)),key=lambda i:(-float(scores[i]),pairs[i][1],pairs[i][0]['event_id']))
            result[name][child['event_id']]=[pairs[i][0]['event_id'] for i in order]
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranker',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory nonempty; refusing overwrite')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    import e5c38_end_to_end as c38
    import e5c5_overlapping_generator as gen
    with a.ranker.open('rb') as f:original=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if original['features']!=base.FEATURES:raise ValueError('Baseline schema mismatch')
    if original['features'][-1]!='gap_rank':raise ValueError('Expected last feature gap_rank')
    if gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Frozen gate incompatible with baseline ranker')
    if float(original['threshold'])!=.65 or int(original['window'])!=90:raise ValueError('Unexpected frozen settings')
    with a.reference.open(newline='',encoding='utf-8-sig') as f:
        refs={(int(r['seed']),float(r['budget_multiplier'])):r for r in csv.DictReader(f)}
    if len(refs)!=25:raise ValueError('Expected 25 development reference rows')
    train_path=a.train/'telemetry.csv';truth_path=a.train/'ground_truth.csv'
    gt={r['event_id']:r['predecessor_event_id'] for r in base.read(truth_path) if r.get('predecessor_event_id')}
    _,_,cands=base.candidates(train_path,original['threshold'],original['window'])
    x=[];y=[];reachable=0
    for child,entries in cands.items():
        parent=gt.get(child)
        if not parent or parent not in {e[0] for e in entries}:continue
        reachable+=1
        for pid,gap,v in entries:x.append(v[:-1]);y.append(int(pid==parent))
    if reachable<10:raise ValueError('Insufficient reachable training edges')
    model=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=0)
    model.fit(np.asarray(x,dtype=float),np.asarray(y,dtype=int))
    print(f'TRAIN ONLY: reachable={reachable} pairs={len(y)} positives={sum(y)}')
    train_events=c38.read(train_path)
    labels={r['event_id']:int(r['malicious']) for r in c38.read(truth_path)}
    feat=c38.extract(train_events,90);ids=[r['event_id'] for r in train_events]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feat[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    rows=[]
    for seed in SEEDS:
        events,truth=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events];features=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([features[i] for i in ids]))[:,1]
        alert_order=[i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))]
        base_n=sum(float(r['signal'])>=.65 for r in events)
        true_parent={r['event_id']:r['predecessor_event_id'] for r in truth if r['predecessor_event_id']}
        for budget in BUDGETS:
            chosen=set(alert_order[:min(len(events),int(round(base_n*budget)))])
            ranks=rank_candidates(events,chosen,original,model,90)
            eligible={child:parent for child,parent in true_parent.items() if parent in ranks['original'].get(child,[])}
            ref=refs[(seed,budget)]
            if len(eligible)!=int(ref['eligible_truth_edges']):raise AssertionError(f'Eligible mismatch seed={seed} budget={budget}')
            orig_correct=sum(ranks['original'][c][0]==parent for c,parent in eligible.items())
            if orig_correct!=int(ref['top1_correct']):raise AssertionError(f'Baseline mismatch seed={seed} budget={budget}')
            for name in ('original','no_gap_rank'):
                correct=sum(ranks[name][c][0]==parent for c,parent in eligible.items())
                mrr=sum(1/(ranks[name][c].index(parent)+1) for c,parent in eligible.items())/len(eligible)
                rows.append(dict(seed=seed,budget_multiplier=budget,variant=name,eligible=len(eligible),top1_correct=correct,top1_accuracy=correct/len(eligible),mrr=mrr))
            print(f'Seed {seed} x{budget:g}: eligible={len(eligible)} baseline={orig_correct} retrained={rows[-1]["top1_correct"]}')
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'per_seed.csv',rows)
    summary=[]
    for budget in BUDGETS:
        for name in ('original','no_gap_rank'):
            group=[r for r in rows if r['budget_multiplier']==budget and r['variant']==name]
            eligible=sum(r['eligible'] for r in group);correct=sum(r['top1_correct'] for r in group)
            summary.append(dict(budget_multiplier=budget,variant=name,eligible=eligible,top1_correct=correct,top1_accuracy=correct/eligible,weighted_mrr=sum(r['mrr']*r['eligible'] for r in group)/eligible))
            print(f'x{budget:g} {name}: {correct}/{eligible} top1={correct/eligible:.4f}')
    write(a.output/'budget_summary.csv',summary)
    obj=dict(model=model,features=base.FEATURES[:-1],training_reachable=reachable,training_pairs=len(y),train_only=True,baseline_sha256=sha(a.ranker),threshold=original['threshold'],window=original['window'])
    with (a.output/'reranker_no_gap_rank.pkl').open('wb') as f:pickle.dump(obj,f)
    report={'experiment':'E5-C56','status':'development ranking audit only','baseline_sha256':sha(a.ranker),'train_telemetry_sha256':sha(train_path),'train_truth_sha256':sha(truth_path),'seeds':list(SEEDS),'budgets':list(BUDGETS),'feature_removed':'gap_rank','gate_reuse':'NOT VALIDATED: original gate trained on baseline score distribution; do not apply directly','independent_test':'NOT RUN; reserve untouched seeds for a separately frozen evaluation'}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved:',a.output.resolve())
if __name__=='__main__':main()
