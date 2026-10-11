"""E5-C32: One-time frozen evaluation on previously sealed synthetic seeds.
No fitting, calibration, threshold search, or ground truth during inference.
Use only trusted local pickle models. Do not rerun after changing the model.
"""
import argparse, csv, hashlib, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def evaluate(rows,truth,keep,events):
    selected=[r for r,k in zip(rows,keep) if k]
    tp=sum((r['parent'],r['child']) in truth for r in selected)
    n=len(selected); fp=n-tp; fn=len(truth)-tp
    p=tp/n if n else 0.0
    recall=tp/len(truth) if truth else 0.0
    return dict(predicted=n,tp=tp,fp=fp,fn=fn,precision=p,recall=recall,
                f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0,
                false_edges_per_1000_events=1000*fp/events if events else 0.0)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--test-root',type=Path,required=True)
    a.add_argument('--scripts',type=Path,required=True)
    a.add_argument('--model',type=Path,required=True)
    a.add_argument('--gate',type=Path,required=True)
    a.add_argument('--baseline-calibration',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    sys.path.insert(0,str(args.scripts.resolve()))
    import e5c15_reranker as base
    gate_module=load_local('e5c31_train_no_parent',args.scripts/'e5c31_train_no_parent.py')
    with args.model.open('rb') as f: ranker=pickle.load(f)
    with args.gate.open('rb') as f: gate=pickle.load(f)
    baseline=json.loads(args.baseline_calibration.read_text(encoding='utf-8'))
    if baseline.get('method')!='e5c17_score_rejection_v1': raise ValueError('Wrong baseline calibration')
    if gate['e5c15_sha256']!=sha256(args.model): raise ValueError('Frozen ranker hash mismatch')
    if gate['features']!=gate_module.FEATURES: raise ValueError('Gate feature mismatch')
    if ranker['features']!=base.FEATURES: raise ValueError('Ranker feature mismatch')
    if (gate['threshold'],gate['window'])!=(ranker['threshold'],ranker['window']): raise ValueError('Ranker configuration mismatch')
    manifest_path=args.test_root/'manifest.json'
    if not manifest_path.is_file(): raise FileNotFoundError('Missing sealed manifest')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    # Enumerate seed directories directly; check manifest content separately for provenance.
    folders=sorted(d for d in args.test_root.iterdir() if d.is_dir() and
                   (d/'telemetry.csv').is_file() and (d/'ground_truth.csv').is_file())
    if len(folders)!=5: raise ValueError(f'Expected exactly 5 test datasets; found {len(folders)}')
    old_cutoff=float(baseline['selected_cutoff'])
    new_cutoff=float(gate['cutoff'])
    results=[]
    totals={'events':0,'truth_edges':0,'baseline_predicted':0,'baseline_tp':0,'gate_predicted':0,'gate_tp':0}
    for folder in folders:
        # Telemetry-only prediction is finished before truth is read.
        rows,audit=gate_module.score_dataset(folder,ranker,base)
        x=np.asarray([r['x'] for r in rows],dtype=float)
        probs=gate['model'].predict_proba(x)[:,1] if len(rows) else np.array([])
        old_keep=[r['score']>=old_cutoff for r in rows]
        new_keep=probs>=new_cutoff
        truth,_=gate_module.read_truth(folder/'ground_truth.csv')
        old=evaluate(rows,truth,old_keep,audit['events'])
        new=evaluate(rows,truth,new_keep,audit['events'])
        results.append({'dataset':folder.name,'events':audit['events'],'truth_edges':len(truth),
                        'candidate_children':len(rows),'baseline':old,'gate':new,
                        'telemetry_sha256':sha256(folder/'telemetry.csv'),
                        'truth_sha256':sha256(folder/'ground_truth.csv')})
        totals['events']+=audit['events'];totals['truth_edges']+=len(truth)
        totals['baseline_predicted']+=old['predicted'];totals['baseline_tp']+=old['tp']
        totals['gate_predicted']+=new['predicted'];totals['gate_tp']+=new['tp']
        print(f"{folder.name}: baseline P={old['precision']:.4f} R={old['recall']:.4f} F1={old['f1']:.4f}; gate P={new['precision']:.4f} R={new['recall']:.4f} F1={new['f1']:.4f}",flush=True)
    def pooled(prefix):
        n=totals[prefix+'_predicted'];tp=totals[prefix+'_tp'];fp=n-tp;fn=totals['truth_edges']-tp
        return dict(predicted=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0.0,
                    recall=tp/totals['truth_edges'] if totals['truth_edges'] else 0.0,
                    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0,
                    false_edges_per_1000_events=1000*fp/totals['events'])
    report={'design':'One-time frozen E5-C17 versus E5-C31 evaluation on five synthetic seeds',
            'manifest_sha256':sha256(manifest_path),'manifest':manifest,
            'ranker_sha256':sha256(args.model),'gate_sha256':sha256(args.gate),
            'baseline_calibration_sha256':sha256(args.baseline_calibration),
            'baseline_cutoff':old_cutoff,'gate_cutoff':new_cutoff,
            'datasets':len(folders),'total_events':totals['events'],'total_truth_edges':totals['truth_edges'],
            'baseline':pooled('baseline'),'gate':pooled('gate'),'per_dataset':results,
            'limitations':['Synthetic correlated events, not verified real causal chains',
            'Historical seed disjointness not independently verified',
            'Test labels used only after frozen predictions for evaluation',
            'Do not tune model or cutoffs using these results']}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'summary.json'
    if path.exists(): raise FileExistsError('Evaluation output already exists; refusing to overwrite sealed test report')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C32 | FROZEN TEST RESULTS')
    print(json.dumps({'datasets':len(folders),'total_events':totals['events'],
                      'total_truth_edges':totals['truth_edges'],'baseline':report['baseline'],
                      'gate':report['gate']},indent=2))
    print('Saved:',path.resolve())

if __name__=='__main__':main()
