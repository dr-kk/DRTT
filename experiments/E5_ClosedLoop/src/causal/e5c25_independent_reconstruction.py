#!/usr/bin/env python3
"""E5-C25 frozen independent reconstruction benchmark on existing E5-C5 datasets.
No model training or cutoff selection on evaluation seeds.
"""
import argparse,csv,json,subprocess,sys,statistics,hashlib
from pathlib import Path

def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def run(cmd):
    cp=subprocess.run(cmd,text=True,capture_output=True)
    if cp.returncode:raise RuntimeError('Command failed: '+ ' '.join(map(str,cmd))+'\n'+cp.stdout+'\n'+cp.stderr)

def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description='Frozen E5-C13 vs E5-C17 independent reconstruction benchmark')
    p.add_argument('--datasets',nargs='+',required=True,help='Directories containing telemetry.csv and ground_truth.csv')
    p.add_argument('--scripts',default='src/causal')
    p.add_argument('--model',required=True,help='Trusted frozen E5-C15 reranker.pkl')
    p.add_argument('--calibration',required=True,help='Frozen E5-C17 calibration.json')
    p.add_argument('--output',required=True)
    a=p.parse_args(); scripts=Path(a.scripts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name in ('e5c13_infer.py','e5c17_rejection.py','e5c17_evaluate.py'):
        if not (scripts/name).is_file():p.error('Missing script: '+str(scripts/name))
    for name in ('model','calibration'):
        if not Path(getattr(a,name)).is_file():p.error('Missing '+name+' file')
    rows=[]; audits=[];seen=set()
    for index,folder in enumerate(map(Path,a.datasets)):
        telemetry=folder/'telemetry.csv';truth=folder/'ground_truth.csv'
        if not telemetry.is_file() or not truth.is_file():p.error('Missing telemetry.csv/ground_truth.csv in '+str(folder))
        tele=read(telemetry);gt=read(truth)
        expected={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
        if not tele or set(tele[0])!=expected:raise ValueError('Unexpected telemetry columns in '+str(folder))
        if len({x['event_id'] for x in tele})!=len(tele):raise ValueError('Duplicate event IDs')
        if {x['event_id'] for x in tele}!={x['event_id'] for x in gt} or len(tele)!=len(gt):raise ValueError('Truth ID mismatch')
        digest=hashlib.sha256(telemetry.read_bytes()).hexdigest()
        if digest in seen:raise ValueError('Duplicate telemetry dataset: '+str(folder))
        seen.add(digest)
        base=out/('dataset_%02d'%index);base.mkdir(exist_ok=True)
        modes={
            'temporal_host': [sys.executable,str(scripts/'e5c13_infer.py'),'--telemetry',str(telemetry),'--output',str(base/'temporal_host')],
            'learned_rejection': [sys.executable,str(scripts/'e5c17_rejection.py'),'infer','--telemetry',str(telemetry),'--model',a.model,'--calibration',a.calibration,'--output',str(base/'learned_rejection')]
        }
        actual={(x['predecessor_event_id'],x['event_id']) for x in gt if x['predecessor_event_id']}
        # Diagnostic oracle: ground-truth edges whose endpoints pass frozen event threshold and are within 90 s.
        byid={x['event_id']:x for x in tele}
        reachable=sum(float(byid[u]['signal'])>=.65 and float(byid[v]['signal'])>=.65 and
                      0<int(byid[v]['timestamp'])-int(byid[u]['timestamp'])<=90 for u,v in actual)
        for method,cmd in modes.items():
            run(cmd)
            pred=base/method
            run([sys.executable,str(scripts/'e5c17_evaluate.py'),'--truth',str(truth),'--predictions',str(pred),'--output',str(pred/'evaluation')])
            report=json.loads((pred/'evaluation'/'evaluation.json').read_text())
            trans=report['transition'];ev=report['event']
            rows.append(dict(dataset=folder.name,method=method,events=len(tele),truth_edges=len(actual),
                             reachable_edges_at_065_window90=reachable,TP=trans['TP'],predicted_edges=trans['predicted'],
                             precision=trans['precision'],recall=trans['recall'],f1=trans['f1'],event_f1=ev['f1']))
        audits.append(dict(dataset=str(folder),telemetry_sha256=digest,events=len(tele),truth_edges=len(actual),
                           reachable_edges_at_065_window90=reachable))
    write(out/'per_dataset.csv',rows)
    summary={'design':'frozen, telemetry-only E5-C13 versus trained E5-C17 with separate evaluation',
             'datasets':audits,'model':a.model,'calibration':a.calibration,
             'methods':{method:{key:{'mean':statistics.mean(r[key] for r in rows if r['method']==method),
                                   'sample_sd':statistics.stdev(r[key] for r in rows if r['method']==method) if len(audits)>1 else None}
                                for key in ('precision','recall','f1','event_f1')}
                        for method in ('temporal_host','learned_rejection')},
             'limitations':['Reuses E5-C5 synthetic overlapping campaigns, not an independently realistic causal dataset',
                            'Frozen learned model and calibration must originate from disjoint development seeds',
                            'No causal-chain feedback to defender; no real network telemetry',
                            'Reachable-edge count is evaluation-only diagnostic and not a rigorous oracle for learned candidate schema']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('E5-C25 | FROZEN INDEPENDENT RECONSTRUCTION BENCHMARK')
    print(json.dumps(summary,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
