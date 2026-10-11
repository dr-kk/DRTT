"""E5-C41 development-only candidate-parent availability diagnostic.
Labels are used for training/evaluation ONLY, never to construct features.
Not an independent test; no frozen E5-C31 artifacts are changed.
"""
import argparse,csv,json,pickle,hashlib,sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,precision_recall_fscore_support

FEATURES=['candidate_count_log','best_gap_norm','mean_gap_norm','min_parent_signal','max_parent_signal','child_signal','best_score','score_margin','score_std','same_source','same_destination','dst_to_src','src_to_dst','same_sensor','same_type']

def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def collect(folder,ranker,base):
    events,selected,cands=base.candidates(folder/'telemetry.csv',ranker['threshold'],ranker['window'])
    gt={r['event_id']:r['predecessor_event_id'] for r in read(folder/'ground_truth.csv')}
    if len(gt)!=len(events):raise ValueError('Ground truth/event count mismatch')
    X=[];y=[];children=[]
    for child,entries in cands.items():
        v=np.asarray([e[2] for e in entries],dtype=float)
        scores=np.asarray(ranker['model'].decision_function(v),dtype=float).reshape(-1)
        order=sorted(range(len(entries)),key=lambda i:(-scores[i],entries[i][1],entries[i][0]))
        top=order[0]; second=order[1] if len(order)>1 else top
        vec=v[top]; gaps=np.asarray([e[1] for e in entries],dtype=float)
        x=[np.log1p(len(entries)),gaps.min()/ranker['window'],gaps.mean()/ranker['window'],
           v[:,7].min(),v[:,7].max(),vec[8],scores[top],scores[top]-scores[second],scores.std(),*vec[1:7]]
        if len(x)!=len(FEATURES):raise ValueError('Feature dimension mismatch')
        true_parent=gt[child]
        available=bool(true_parent) and any(e[0]==true_parent for e in entries)
        X.append(x);y.append(int(available));children.append(child)
    return np.asarray(X,dtype=float),np.asarray(y,dtype=int),children,{'events':len(events),'selected':len(selected),'candidate_children':len(X),'available':int(sum(y))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    p.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    p.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    p.add_argument('--scripts',type=Path,default=Path('src/causal'))
    p.add_argument('--output',type=Path,default=Path('results/causal/e5c41'))
    a=p.parse_args()
    if a.train.resolve()==a.calibration.resolve():raise ValueError('Train/calibration must differ')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c15_reranker as base
    with a.ranker.open('rb') as f:ranker=pickle.load(f) # trusted local artifact
    if ranker['features']!=base.FEATURES:raise ValueError('Ranker feature mismatch')
    X,y,_,ta=collect(a.train,ranker,base)
    V,z,ids,ca=collect(a.calibration,ranker,base)
    if len(set(y))!=2 or len(set(z))!=2:raise ValueError('Both classes required')
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=0))
    model.fit(X,y)
    probs=model.predict_proba(V)[:,1]
    pred=(probs>=0.5).astype(int)
    prec,rec,f1,_=precision_recall_fscore_support(z,pred,average='binary',zero_division=0)
    report={'experiment':'E5-C41','scope':'development-only exploratory candidate availability, NOT independent validation',
            'target':'actual predecessor exists among observable candidates (NOT top1 correctness)',
            'train':ta,'calibration':ca,'features':FEATURES,'threshold':0.5,
            'calibration_metrics':{'auroc':float(roc_auc_score(z,probs)),'average_precision':float(average_precision_score(z,probs)),
            'log_loss':float(log_loss(z,probs)),'precision_at_0_5':float(prec),'recall_at_0_5':float(rec),'f1_at_0_5':float(f1),
            'predicted_available':int(pred.sum())},
            'sha256':{'ranker':digest(a.ranker),'train_telemetry':digest(a.train/'telemetry.csv'),
                      'calibration_telemetry':digest(a.calibration/'telemetry.csv')},
            'limitations':['Calibration reused by earlier experiments; exploratory only','Same synthetic generator','No model selection on E5-C39/C40 data','No end-to-end improvement established']}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'calibration_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['child_event_id','true_parent_available','predicted_probability','predicted_available_at_0_5'])
        w.writerows((i,int(yi),float(pi),int(pr)) for i,yi,pi,pr in zip(ids,z,probs,pred))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.output/'candidate_validity_model.pkl').open('wb') as f:pickle.dump({'model':model,'features':FEATURES,'ranker_sha256':digest(a.ranker),'threshold':0.5},f)
    print('E5-C41 | DEVELOPMENT-ONLY CANDIDATE VALIDITY')
    print('Train:',ta,'Calibration:',ca)
    print(json.dumps(report['calibration_metrics'],indent=2))
    print('WARNING: exploratory calibration only; do not claim independent validation.')
    print('Saved:',a.output.resolve())
if __name__=='__main__':main()
