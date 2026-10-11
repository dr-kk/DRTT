"""E5-C44: frozen E5-C43 models on newly generated synthetic seeds.
No fitting, calibration, threshold selection, or test-label use for predictions.
NOTE: seeds are newly specified, but historical nonuse cannot be verified.
"""
import argparse,csv,json,pickle,hashlib,sys,tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def ece(y,p):
    edges=np.linspace(0,1,11);out=0.
    for k in range(10):
        m=(p>=edges[k])&((p<=edges[k+1]) if k==9 else (p<edges[k+1]))
        if m.any():out+=float(m.mean()*abs(float(y[m].mean())-float(p[m].mean())))
    return out

def evaluate(y,p):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),mean_probability=float(p.mean()),
        auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),ece_10=ece(y,p),
        tp=tp,fp=fp,fn=fn,tn=tn,precision_0_5=tp/(tp+fp) if tp+fp else 0.,
        recall_0_5=tp/(tp+fn) if tp+fn else 0.)

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=Path,default=Path('results/causal/e5c43/models.pkl'))
    ap.add_argument('--ranker',type=Path,default=Path('results/causal/e5c15/reranker.pkl'))
    ap.add_argument('--scripts',type=Path,default=Path('src/causal'))
    ap.add_argument('--output',type=Path,default=Path('results/causal/e5c44'))
    ap.add_argument('--start-seed',type=int,default=4400)
    ap.add_argument('--num-seeds',type=int,default=5)
    a=ap.parse_args()
    if a.num_seeds<2:ap.error('At least 2 seeds required')
    if a.start_seed<4000:ap.error('Use seed range >=4000 to avoid known development seeds')
    if a.output.exists() and any(a.output.iterdir()):
        ap.error('Output directory is nonempty: use a fresh output path to preserve prior results')
    sys.path.insert(0,str(a.scripts.resolve()))
    import e5c41_candidate_validity as c41
    import e5c15_reranker as base
    import e5c5_overlapping_generator as gen
    with a.models.open('rb') as f:obj=pickle.load(f) # trusted local artifact only
    with a.ranker.open('rb') as f:ranker=pickle.load(f)
    if obj['features']!=c41.FEATURES or obj['ranker_sha256']!=sha(a.ranker):raise ValueError('Model feature/ranker provenance mismatch')
    if ranker['features']!=base.FEATURES:raise ValueError('Reranker feature mismatch')
    models=obj['models']
    expected={'balanced','unweighted','balanced_sigmoid_cv5'}
    if set(models)!=expected:raise ValueError('Unexpected frozen models')
    a.output.mkdir(parents=True,exist_ok=True)
    seed_rows=[];pred_rows=[];manifest=[]
    for seed in range(a.start_seed,a.start_seed+a.num_seeds):
        events,truth=gen.generate(seed,100,2400,10000)
        with tempfile.TemporaryDirectory(prefix='e5c44_') as td:
            folder=Path(td)
            gen.write_csv(folder/'telemetry.csv',events,list(events[0]))
            gen.write_csv(folder/'ground_truth.csv',truth,list(truth[0]))
            X,y,ids,info=c41.collect(folder,ranker,base)
            manifest.append({'seed':seed,'telemetry_sha256':sha(folder/'telemetry.csv'),
                             'truth_sha256':sha(folder/'ground_truth.csv'),**info})
        if len(set(y.tolist()))!=2:raise RuntimeError(f'Seed {seed} lacks both labels')
        for name,model in models.items():
            p=np.asarray(model.predict_proba(X)[:,1],dtype=float)
            m=evaluate(y,p)
            seed_rows.append({'seed':seed,'model':name,**m})
            pred_rows.extend({'seed':seed,'model':name,'child_event_id':i,'true_parent_available':int(yi),
                              'predicted_probability':float(pi),'predicted_at_0_5':int(pi>=.5)}
                             for i,yi,pi in zip(ids,y,p))
            print(f'Seed {seed} {name}: AUROC={m["auroc"]:.4f} AP={m["average_precision"]:.4f} '
                  f'Brier={m["brier"]:.4f} ECE={m["ece_10"]:.4f} TP/FP={m["tp"]}/{m["fp"]}')
    write_csv(a.output/'per_seed.csv',seed_rows)
    write_csv(a.output/'predictions.csv',pred_rows)
    summary={}
    for name in sorted(models):
        rr=[r for r in seed_rows if r['model']==name]
        pooled=[r for r in pred_rows if r['model']==name]
        y=np.array([r['true_parent_available'] for r in pooled],dtype=int)
        p=np.array([r['predicted_probability'] for r in pooled],dtype=float)
        summary[name]={'pooled':evaluate(y,p),'mean_per_seed':{k:float(np.mean([r[k] for r in rr])) for k in
            ('auroc','average_precision','brier','log_loss','ece_10','precision_0_5','recall_0_5')}}
    base_y=np.array([r['true_parent_available'] for r in pred_rows if r['model']=='unweighted'],dtype=int)
    prevalence=float(base_y.mean())
    report={'experiment':'E5-C44','scope':'frozen models, fresh specified synthetic seeds; historical nonuse unverified',
        'seed_range':[a.start_seed,a.start_seed+a.num_seeds-1],
        'generator':{'campaigns':100,'duration':2400,'background_events':10000},
        'models_sha256':sha(a.models),'ranker_sha256':sha(a.ranker),
        'constant_test_prevalence_brier_reference_only':prevalence*(1-prevalence),
        'summary':summary,'seed_manifest':manifest,
        'limitations':['Same synthetic generator as development; not external validity',
                       'Seeds specified after development; historical nonuse not independently verified',
                       'No thresholds tuned; fixed 0.50 decisions are not end-to-end causal edges',
                       'Test prevalence baseline is descriptive, not an operational deployable baseline']}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nE5-C44 | FROZEN SYNTHETIC SEED EVALUATION')
    for name,metrics in summary.items():
        m=metrics['pooled']
        print(f'{name}: n={m["n"]} prevalence={m["prevalence"]:.4%} AUROC={m["auroc"]:.4f} '
              f'AP={m["average_precision"]:.4f} Brier={m["brier"]:.4f} '
              f'logloss={m["log_loss"]:.4f} ECE={m["ece_10"]:.4f} '
              f'meanP={m["mean_probability"]:.4f} TP/FP={m["tp"]}/{m["fp"]}')
    print('WARNING: synthetic-only candidate validity; no end-to-end improvement established.')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
