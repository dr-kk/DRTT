"""E5-C49: development-only joint edge-correctness diagnostic.
Train on E5-C15 signal-selected candidates; evaluate on E5-C17 context-selected
candidates. No test seeds, threshold selection, or end-to-end claims.
"""
import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def metrics(y,p):
    if len(set(y))!=2: raise ValueError('Both classes required')
    return {'n':len(y),'positives':int(sum(y)),'prevalence':float(np.mean(y)),
        'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
        'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
        'mean_probability':float(np.mean(p))}

def load_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=Path,default=Path('results/causal/e5c43/models.pkl'))
    ap.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--gate',type=Path,default=Path('results/causal/e5c31/no_parent_gate.pkl'))
    ap.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    ap.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,default=Path('results/causal/e5c49'))
    a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error('Output must be empty; prior results preserved')
    if a.train.resolve()==a.calibration.resolve():ap.error('Train and calibration must differ')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c45_end_to_end_candidate_filter as c45
    import e5c38_end_to_end as c38
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    with a.models.open('rb') as f: bundle=pickle.load(f)
    with a.ranker.open('rb') as f: ranker=pickle.load(f)
    with a.gate.open('rb') as f: gate=pickle.load(f)
    if bundle['ranker_sha256']!=sha(a.ranker) or gate['e5c15_sha256']!=sha(a.ranker):raise ValueError('Ranker hash mismatch')
    if bundle['features']!=c41.FEATURES or gate['features']!=c38.GATE_FEATURES or ranker['features']!=base.FEATURES:raise ValueError('Feature mismatch')
    if float(ranker['threshold'])!=.65 or int(ranker['window'])!=90 or int(gate['window'])!=90:raise ValueError('Unexpected frozen settings')
    def assemble(folder,selection,detector=None):
        events=c38.read(folder/'telemetry.csv')
        truthrows=load_rows(folder/'ground_truth.csv')
        truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
        if selection=='signal':
            chosen={r['event_id'] for r in events if float(r['signal'])>=.65}
        else:
            feat=c38.extract(events,90);ids=[r['event_id'] for r in events]
            prob=detector.predict_proba(np.asarray([feat[i] for i in ids]))[:,1]
            n=sum(float(r['signal'])>=.65 for r in events)
            chosen={i for i,_ in sorted(zip(ids,prob),key=lambda z:(-float(z[1]),z[0]))[:n]}
        rec=c45.candidates_with_features(events,chosen,ranker,90)
        X=np.asarray([r['valid_features']+r['gate_features'] for r in rec],dtype=float)
        y=np.asarray([int((r['parent'],r['child']) in truth) for r in rec],dtype=int)
        G=np.asarray([r['gate_features'] for r in rec],dtype=float)
        V=np.asarray([r['valid_features'] for r in rec],dtype=float)
        return rec,X,y,G,V,len(events),len(chosen)
    trainrows=c38.read(a.train/'telemetry.csv')
    traintruth={r['event_id']:int(r['malicious']) for r in load_rows(a.train/'ground_truth.csv')}
    trfeat=c38.extract(trainrows,90);trids=[r['event_id'] for r in trainrows]
    detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
    detector.fit(np.asarray([trfeat[i] for i in trids]),np.asarray([traintruth[i] for i in trids]))
    tr,X,y,G,V,_,_=assemble(a.train,'signal')
    if len(set(y))!=2:raise ValueError('Train missing class')
    joint=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight=None,random_state=0))
    joint.fit(X,y)
    rec,Z,z,H,W,ne,alerts=assemble(a.calibration,'context',detector)
    gp=gate['model'].predict_proba(H)[:,1]
    vp=bundle['models']['unweighted'].predict_proba(W)[:,1]
    jp=joint.predict_proba(Z)[:,1]
    # Product is a diagnostic ranking score, not a calibrated probability.
    product=gp*vp
    report={'experiment':'E5-C49','scope':'development-only; E5-C15 signal-selected train, E5-C17 context-selected evaluation',
        'train':{'candidate_children':len(y),'correct_top1':int(sum(y))},
        'calibration':{'events':ne,'alerts':alerts,'candidate_children':len(z),'correct_top1':int(sum(z))},
        'scores':{'existing_gate':metrics(z,gp),'joint_logistic':metrics(z,jp),
                  'unweighted_validity_probability':metrics(z,vp),
                  'gate_times_validity_rank_score':{'auroc':float(roc_auc_score(z,product)),
                    'average_precision':float(average_precision_score(z,product))}},
        'feature_order':c41.FEATURES+c38.GATE_FEATURES,
        'sha256':{'models':sha(a.models),'ranker':sha(a.ranker),'gate':sha(a.gate),
                  'train_telemetry':sha(a.train/'telemetry.csv'), 'calibration_telemetry':sha(a.calibration/'telemetry.csv')},
        'limitations':['E5-C17 reused for development; not independent test',
          'Signal-selected training candidates versus context-selected evaluation candidates',
          'Some features overlap between validity and gate feature sets',
          'Random events/campaigns may be correlated within a synthetic seed',
          'No cutoff selection or end-to-end improvement claim',
          'Product score is not asserted to be a calibrated probability']}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'development_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['child','parent','correct_top1','gate_probability','validity_probability','joint_probability','product_rank_score'])
        w.writerows((r['child'],r['parent'],int(yi),float(g),float(v),float(j),float(p)) for r,yi,g,v,j,p in zip(rec,z,gp,vp,jp,product))
    with (a.output/'joint_model.pkl').open('wb') as f:pickle.dump({'model':joint,'features':report['feature_order'],'sha256':report['sha256']},f)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C49 | DEVELOPMENT-ONLY JOINT EDGE CORRECTNESS')
    print('Train top1 correct:',int(sum(y)),'/',len(y),'Calibration:',int(sum(z)),'/',len(z))
    for name,vals in report['scores'].items():
        print(f'{name}: AUROC={vals["auroc"]:.4f} AP={vals["average_precision"]:.4f}'+
              (f' Brier={vals["brier"]:.4f} logloss={vals["log_loss"]:.4f}' if 'brier' in vals else ''))
    print('WARNING: development-only; do not claim independent improvement or tune on E5-C47/C48.')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
