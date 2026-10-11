"""E5-C48: post-hoc false-rejection audit of frozen E5-C47, seeds 4700-4704.
No tuning, model fitting beyond the original E5-C38 detector, or threshold selection.
"""
import argparse,csv,json,pickle,sys,hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(vals):
    return {'n':len(vals),'mean':float(np.mean(vals)) if vals else None,
            'median':float(np.median(vals)) if vals else None,
            'q10':float(np.quantile(vals,.1)) if vals else None,
            'q90':float(np.quantile(vals,.9)) if vals else None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,default=Path('results/causal/e5c46/selection.json'))
    p.add_argument('--models',type=Path,default=Path('results/causal/e5c43/models.pkl'))
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    p.add_argument('--reference',type=Path,default=Path('results/causal/e5c47/per_seed.csv'))
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c48'))
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory must be empty')
    sel=json.loads(a.selection.read_text(encoding='utf-8'))
    if sel['experiment']!='E5-C46' or sel['selected_model']!='unweighted':raise ValueError('Selection mismatch')
    cutoff=float(sel['selected_threshold'])
    for key,path in [('models',a.models),('ranker',a.ranker),('gate',a.gate),('train_telemetry',a.train/'telemetry.csv')]:
        if sel['sha256'][key]!=sha(path):raise ValueError(f'{key} hash mismatch')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.models.open('rb') as f:bundle=pickle.load(f)
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    with a.gate.open('rb') as f:gate=pickle.load(f)
    if bundle['ranker_sha256']!=sha(a.ranker) or gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Ranker mismatch')
    if bundle['features']!=c41.FEATURES or gate['features']!=c38.GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature mismatch')
    model=bundle['models']['unweighted']
    train=c38.read(a.train/'telemetry.csv')
    labels={r['event_id']:int(r['malicious']) for r in c38.read(a.train/'ground_truth.csv')}
    feats=c38.extract(train,90)
    ids=[r['event_id'] for r in train]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([labels[i] for i in ids]))
    with a.reference.open(newline='',encoding='utf-8-sig') as f:ref={(int(r['seed']),r['pipeline']):r for r in csv.DictReader(f)}
    allrows=[];per=[]
    for seed in range(4700,4705):
        events,truthrows=gen.generate(seed,100,2400,10000)
        ids=[r['event_id'] for r in events]
        x=c38.extract(events,90)
        probs=detector.predict_proba(np.asarray([x[i] for i in ids]))[:,1]
        n=sum(float(r['signal'])>=.65 for r in events)
        chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
        records=c45.candidates_with_features(events,chosen,ranker,90)
        expected,audit=c38.top_predictions(events,chosen,ranker,gate,base,90)
        if len(records)!=audit['candidate_children']:raise AssertionError('Candidate count mismatch')
        g=gate['model'].predict_proba(np.asarray([r['gate_features'] for r in records]))[:,1] if records else []
        v=model.predict_proba(np.asarray([r['valid_features'] for r in records]))[:,1] if records else []
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        baseline=set();filtered=set();categories={'retained_correct':0,'rejected_correct':0,'retained_false':0,'rejected_false':0}
        for r,gp,vp in zip(records,g,v):
            if gp<float(gate['cutoff']):continue
            edge=(r['parent'],r['child']);baseline.add(edge)
            correct=edge in truth;retained=float(vp)>=cutoff
            if retained:filtered.add(edge)
            category=('retained_' if retained else 'rejected_')+('correct' if correct else 'false')
            categories[category]+=1
            record={'seed':seed,'parent':r['parent'],'child':r['child'],'category':category,
                    'correct_edge':int(correct),'retained':int(retained),'validity_probability':float(vp),
                    'gate_probability':float(gp),'validity_threshold':cutoff}
            record.update({name:float(val) for name,val in zip(c41.FEATURES,r['valid_features'])})
            allrows.append(record)
        if baseline!=set(expected):raise AssertionError(f'Seed {seed}: baseline mismatch')
        b=c45.evaluate(baseline,truth,len(events));f=c45.evaluate(filtered,truth,len(events))
        for name,metrics in [('context_frozen_gate',b),('context_unweighted_frozen_e5c46',f)]:
            key=(seed,name)
            if key not in ref:raise AssertionError(f'Missing reference {key}')
            for metric in ('tp','fp','predicted'):
                if int(ref[key][metric])!=metrics[metric]:raise AssertionError(f'Seed {seed} {name} {metric}: {metrics[metric]} != {ref[key][metric]}')
        per.append({'seed':seed,**categories,'baseline_tp':b['tp'],'baseline_fp':b['fp'],'filtered_tp':f['tp'],'filtered_fp':f['fp']})
        print(f'Seed {seed}: rejected correct={categories["rejected_correct"]}, rejected false={categories["rejected_false"]}, retained correct={categories["retained_correct"]}, retained false={categories["retained_false"]}')
    cats=('retained_correct','rejected_correct','retained_false','rejected_false')
    total={k:sum(row[k] for row in per) for k in cats}
    distribution={k:stats([r['validity_probability'] for r in allrows if r['category']==k]) for k in cats}
    report={'experiment':'E5-C48','scope':'post-hoc diagnostic on previously evaluated E5-C47 seeds; no tuning',
            'threshold':cutoff,'counts':total,'probability_distribution':distribution,
            'rejected_false_fraction':total['rejected_false']/(total['rejected_false']+total['rejected_correct']),
            'true_edge_retention':total['retained_correct']/(total['retained_correct']+total['rejected_correct']),
            'sha256':{'selection':sha(a.selection),'models':sha(a.models),'ranker':sha(a.ranker),'gate':sha(a.gate),'reference':sha(a.reference)},
            'limitations':['Same synthetic generator','Test seeds 4700-4704 already inspected: diagnostic only','Do not select thresholds from these results','No real-world forensic validation']}
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'accepted_edge_diagnostics.csv',allrows)
    write(a.output/'per_seed_audit.csv',per)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C48 | FALSE-REJECTION AUDIT')
    print(json.dumps({'counts':total,'probability_distribution':distribution,'true_edge_retention':report['true_edge_retention']},indent=2))
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
