"""E5-C43: development-only class-weighting and cross-validated calibration comparison.
Uses E5-C15 train to fit models and 5-fold CV sigmoid calibrator; E5-C17
for exploratory comparison ONLY. No use of E5-C38/39/40 test data.
"""
import argparse, csv, json, pickle, sys, hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score,average_precision_score,log_loss,brier_score_loss,
                             precision_recall_fscore_support,confusion_matrix)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); result=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & ((p<=edges[i+1]) if i==bins-1 else (p<edges[i+1]))
        if mask.any():result+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(result)

def evaluate(y,p):
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return dict(auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
                ece_10=float(ece(y,p)),mean_probability=float(p.mean()),
                precision_at_0_5=float(precision),recall_at_0_5=float(recall),f1_at_0_5=float(f1),
                tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))

def make_lr(weight):
    return make_pipeline(StandardScaler(),LogisticRegression(class_weight=weight,max_iter=3000,random_state=0))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
    a.add_argument('--calibration',type=Path,default=Path('data/causal/e5c17_calibration'))
    a.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    a.add_argument('--scripts',type=Path,default=Path('src/causal'))
    a.add_argument('--output',type=Path,default=Path('results/causal/e5c43'))
    x=a.parse_args()
    if x.train.resolve()==x.calibration.resolve():raise ValueError('Train and calibration folders must differ')
    sys.path.insert(0,str(x.scripts.resolve()))
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    with x.ranker.open('rb') as f: ranker=pickle.load(f) # trusted local file only
    if ranker['features']!=base.FEATURES:raise ValueError('Ranker feature mismatch')
    X,y,_,train_info=c41.collect(x.train,ranker,base)
    V,z,ids,cal_info=c41.collect(x.calibration,ranker,base)
    if len(np.unique(y))!=2 or len(np.unique(z))!=2:raise ValueError('Need both classes')
    if len(X)!=train_info['candidate_children'] or len(V)!=cal_info['candidate_children']:raise AssertionError('Row mismatch')
    models={'balanced':make_lr('balanced'),'unweighted':make_lr(None)}
    models['balanced_sigmoid_cv5']=CalibratedClassifierCV(estimator=make_lr('balanced'),method='sigmoid',
                              cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=43))
    preds={}; metrics={}
    for name,model in models.items():
        model.fit(X,y)
        probs=np.asarray(model.predict_proba(V)[:,1],dtype=float)
        preds[name]=probs
        metrics[name]=evaluate(z,probs)
        print(f'{name}: AUROC={metrics[name]["auroc"]:.4f} AP={metrics[name]["average_precision"]:.4f} '
              f'Brier={metrics[name]["brier"]:.4f} logloss={metrics[name]["log_loss"]:.4f} '
              f'ECE={metrics[name]["ece_10"]:.4f} meanP={metrics[name]["mean_probability"]:.4f} '
              f'TP/FP={metrics[name]["tp"]}/{metrics[name]["fp"]}')
    x.output.mkdir(parents=True,exist_ok=True)
    with (x.output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);names=list(models)
        w.writerow(['child_event_id','true_parent_available']+[f'p_{n}' for n in names])
        for i,ident in enumerate(ids):w.writerow([ident,int(z[i])]+[float(preds[n][i]) for n in names])
    base_rate=float(z.mean()); baseline=float(brier_score_loss(z,np.full(len(z),base_rate)))
    summary={'experiment':'E5-C43','scope':'DEVELOPMENT ONLY; E5-C17 reused in earlier experiments; no independent generalization claim',
      'method':'Balanced and unweighted LR fit E5-C15; sigmoid calibrated balanced LR with stratified 5-fold CV entirely within E5-C15',
      'train':train_info,'calibration':cal_info,'train_prevalence':float(y.mean()),'calibration_prevalence':base_rate,
      'calibration_base_rate_brier':baseline,'models':metrics,'features':c41.FEATURES,
      'sha256':{'ranker':sha(x.ranker),'train_telemetry':sha(x.train/'telemetry.csv'),'calibration_telemetry':sha(x.calibration/'telemetry.csv')},
      'limitations':['Synthetic generator only','E5-C17 already used for model development','No threshold optimization','No frozen end-to-end test','CV folds are event-level and may share campaigns; not independent campaign validation']}
    (x.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (x.output/'models.pkl').open('wb') as f:pickle.dump({'models':models,'features':c41.FEATURES,'ranker_sha256':sha(x.ranker)},f)
    print(f'Observed prevalence={base_rate:.4%}; constant base-rate Brier={baseline:.4f}')
    print('WARNING: exploratory development comparison, not independent validation.')
    print('Saved:',x.output.resolve())
if __name__=='__main__':main()
