import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

# Import the exact feature computation used in E5-C36.
from e5c36_context_incremental import features
from e5c5_overlapping_generator import generate

def save_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description='E5-C37 fresh-seed synthetic context validation; no test tuning')
    p.add_argument('--train',default='data/causal/e5c15_train')
    p.add_argument('--output',default='results/causal/e5c37')
    p.add_argument('--window',type=int,default=90)
    p.add_argument('--seeds',type=int,nargs='+',default=[3700,3701,3702,3703,3704])
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Duplicate seeds')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    X,y=features(Path(a.train),a.window)
    if set(np.unique(y))!={0,1}:raise ValueError('Training labels must contain both classes')
    models={}
    for name,cols in [('signal_only',[0]),('signal_plus_context',[0,1,2,3])]:
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
        m.fit(X[:,cols],y);models[name]=(m,cols)
    rows=[];hashes=[]
    for seed in a.seeds:
        d=out/'generated_data'/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
        telemetry,truth=generate(seed,100,2400,10000)
        save_csv(d/'telemetry.csv',telemetry,['event_id','timestamp','source_host','destination_host','sensor','event_type','signal'])
        save_csv(d/'ground_truth.csv',truth,['event_id','malicious','stage','campaign_id','predecessor_event_id'])
        V,z=features(d,a.window)
        record={'seed':seed,'events':len(z),'malicious':int(z.sum())}
        for name,(m,cols) in models.items():
            scores=m.predict_proba(V[:,cols])[:,1]
            record.update({f'{name}_auroc':float(roc_auc_score(z,scores)),f'{name}_ap':float(average_precision_score(z,scores)),f'{name}_logloss':float(log_loss(z,scores))})
        for metric in ('auroc','ap','logloss'):
            record[f'delta_{metric}']=record[f'signal_plus_context_{metric}']-record[f'signal_only_{metric}']
        rows.append(record)
        hashes.append({'seed':seed,'telemetry_sha256':sha(d/'telemetry.csv'),'ground_truth_sha256':sha(d/'ground_truth.csv')})
    save_csv(out/'per_seed.csv',rows,list(rows[0]))
    summary={'experiment':'E5-C37','scope':'fresh synthetic seeds from same generator; no independent operational validation','train':str(Path(a.train).resolve()),'train_telemetry_sha256':sha(Path(a.train)/'telemetry.csv'),'train_ground_truth_sha256':sha(Path(a.train)/'ground_truth.csv'),'generator_parameters':{'campaigns':100,'duration':2400,'background_events':10000},'window':a.window,'seeds':a.seeds,'data_hashes':hashes,'metrics':{}}
    for metric in ('auroc','ap','logloss'):
        dif=np.array([r[f'delta_{metric}'] for r in rows],dtype=float)
        summary['metrics'][metric]={'signal_only_mean':float(np.mean([r[f'signal_only_{metric}'] for r in rows])),'signal_plus_context_mean':float(np.mean([r[f'signal_plus_context_{metric}'] for r in rows])),'paired_mean_difference':float(dif.mean()),'paired_sd':float(dif.std(ddof=1)) if len(dif)>1 else None,'per_seed_differences':dif.tolist()}
    summary['limitations']=['Five realizations share one synthetic generator','Seed non-overlap with historical runs is not independently established','No threshold selection, feature selection, or model retraining using test labels','Previously explored features and model choices on development data']
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C37 | FRESH SYNTHETIC SEED EVALUATION (NO TEST TUNING)')
    for metric in ('auroc','ap','logloss'):
        v=summary['metrics'][metric];print(f"{metric:>8}: signal={v['signal_only_mean']:.4f} context={v['signal_plus_context_mean']:.4f} paired delta={v['paired_mean_difference']:+.4f}")
    print('Saved:',out.resolve())
    print('WARNING: synthetic generalization only; test labels used exclusively for evaluation.')
if __name__=='__main__':main()
