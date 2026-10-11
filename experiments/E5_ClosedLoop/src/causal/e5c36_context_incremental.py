import argparse,csv,json,random
from pathlib import Path
from collections import deque,defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def read(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def features(folder,window):
    ev=read(folder/'telemetry.csv'); gt={r['event_id']:int(r['malicious']) for r in read(folder/'ground_truth.csv')}
    assert len(gt)==len(ev) and all(r['event_id'] in gt for r in ev)
    # Causal time batches: events sharing a timestamp cannot observe one another.
    ev.sort(key=lambda r:(int(r['timestamp']),r['event_id']))
    history=deque(); src=defaultdict(int); dst=defaultdict(int); pair=defaultdict(int)
    rows=[]; i=0
    while i<len(ev):
        t=int(ev[i]['timestamp'])
        while history and history[0][0]<t-window:
            _,s,d=history.popleft();src[s]-=1;dst[d]-=1;pair[(s,d)]-=1
        j=i
        while j<len(ev) and int(ev[j]['timestamp'])==t:j+=1
        for r in ev[i:j]:
            s,d=r['source_host'],r['destination_host']
            rows.append(([float(r['signal']),src[s],dst[d],pair[(s,d)]],gt[r['event_id']]))
        for r in ev[i:j]:
            s,d=r['source_host'],r['destination_host'];history.append((t,s,d));src[s]+=1;dst[d]+=1;pair[(s,d)]+=1
        i=j
    return np.array([r[0] for r in rows]),np.array([r[1] for r in rows])

def main():
    p=argparse.ArgumentParser(description='E5-C36 development-only incremental context validation')
    p.add_argument('--train',default='data/causal/e5c15_train')
    p.add_argument('--validation',default='data/causal/e5c17_calibration')
    p.add_argument('--output',default='results/causal/e5c36')
    p.add_argument('--window',type=int,default=90)
    a=p.parse_args();tr=Path(a.train);va=Path(a.validation)
    if tr.resolve()==va.resolve():raise ValueError('Training and validation folders must differ')
    X,y=features(tr,a.window);V,z=features(va,a.window)
    if len(set(y))<2 or len(set(z))<2:raise ValueError('Both datasets need both classes')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'experiment':'E5-C36','scope':'development-only; exploratory; validation previously used in E5-C17/E5-C31','train':str(tr),'validation':str(va),'window':a.window,'train_events':len(y),'validation_events':len(z),'models':{}}
    for name,cols in [('signal_only',[0]),('signal_plus_context',[0,1,2,3])]:
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
        model.fit(X[:,cols],y)
        scores=model.predict_proba(V[:,cols])[:,1]
        result['models'][name]={'roc_auc':float(roc_auc_score(z,scores)),'average_precision':float(average_precision_score(z,scores)),'log_loss':float(log_loss(z,scores))}
    s=result['models'];result['delta_context_minus_signal']={k:s['signal_plus_context'][k]-s['signal_only'][k] for k in ('roc_auc','average_precision','log_loss')}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E5-C36 | DEVELOPMENT-ONLY INCREMENTAL CONTEXT TEST')
    for name,m in s.items():print(f"{name:>21}: AUROC={m['roc_auc']:.4f} AP={m['average_precision']:.4f} logloss={m['log_loss']:.4f}")
    print('Context minus signal:',json.dumps(result['delta_context_minus_signal']))
    print('WARNING: exploratory validation, NOT independent held-out evidence.')
    print('Saved:',out.resolve())
if __name__=='__main__':main()
