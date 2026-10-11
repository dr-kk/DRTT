#!/usr/bin/env python3
"""E5-C20: synthetic evidence artifact provenance + blind transition reconstruction.
Consumes E5-C19 trace; does not modify it. Not a real chain-of-custody system.
"""
import argparse,csv,hashlib,json,random,statistics
from pathlib import Path

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(obj):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()
def readcsv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(path,rows,cols):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def build(traces,seed,drop_rate,tamper_rate):
    artifacts=[]; custody=[]; truth=[]
    for r in traces:
        if r['action']=='none':continue
        episode=int(r['episode']);step=int(r['step']);policy=r['policy']
        event_id=f'{policy}:E{episode:05d}:S{step:02d}'
        prior=f'{policy}:E{episode:05d}:S{step-1:02d}' if step else ''
        # The simulator's event order is ground truth. Never include hidden path/progress in defender-facing evidence.
        truth.append({'event_id':event_id,'predecessor_event_id':prior,'policy':policy,'episode':episode})
        rng=random.Random(seed*100003+episode*919+step*67+sum(map(ord,policy)))
        if rng.random()<drop_rate:continue
        payload={'event_id':event_id,'episode':episode,'step':step,'policy':policy,
                 'observation':r['observation'],'action':r['action']}
        original=sha(payload)
        stored=dict(payload)
        tampered=rng.random()<tamper_rate
        if tampered:stored['action']='TAMPERED_'+stored['action']
        artifacts.append({'event_id':event_id,'policy':policy,'episode':episode,'step':step,
                          'payload_json':canonical(stored),'sealed_sha256':original,
                          'stored_sha256':sha(stored)})
        custody.append({'event_id':event_id,'operation':'capture','record_sha256':original})
        custody.append({'event_id':event_id,'operation':'verification','record_sha256':sha(stored)})
    return artifacts,custody,truth

def audit(artifacts,truth):
    valid=[];invalid=[]
    for r in artifacts:
        payload=json.loads(r['payload_json'])
        # Integrity test uses the original sealed hash; no truth needed.
        if sha(payload)==r['sealed_sha256'] and r['stored_sha256']==sha(payload):valid.append(r)
        else:invalid.append(r)
    # Blind reconstruction: predecessor must be the immediately prior observed step in same episode.
    # This intentionally conservative baseline cannot infer links over missing/corrupted evidence.
    groups={}
    for r in valid:groups.setdefault((r['policy'],r['episode']),[]).append(r)
    predicted=[]
    for (policy,episode),rows in groups.items():
        rows.sort(key=lambda x:int(x['step']))
        bystep={int(r['step']):r['event_id'] for r in rows}
        for step,child in bystep.items():
            if step-1 in bystep:predicted.append({'parent_event_id':bystep[step-1],'child_event_id':child})
    gt={(r['predecessor_event_id'],r['event_id']) for r in truth if r['predecessor_event_id']}
    pred={(r['parent_event_id'],r['child_event_id']) for r in predicted}
    tp=len(gt&pred); precision=tp/len(pred) if pred else 0;recall=tp/len(gt) if gt else 0
    return predicted,{'artifact_count':len(artifacts),'integrity_valid':len(valid),
      'integrity_invalid':len(invalid),'truth_event_count':len(truth),
      'evidence_coverage':len(artifacts)/len(truth) if truth else 0,
      'verified_coverage':len(valid)/len(truth) if truth else 0,
      'transition':{'TP':tp,'predicted':len(pred),'ground_truth':len(gt),'precision':precision,
                    'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0},
      'notes':['Ground-truth ordering derived from synthetic simulator trace, not real attacks',
               'Sealed hashes assume trustworthy capture; no external timestamp or independent custody authority',
               'Reconstruction uses sequence indices and is a deliberately simple baseline',
               'No causal inference or evidence-driven defender feedback is established']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=2051);p.add_argument('--drop-rate',type=float,default=.1)
    p.add_argument('--tamper-rate',type=float,default=.05)
    a=p.parse_args()
    if not 0<=a.drop_rate<=1 or not 0<=a.tamper_rate<=1:p.error('rates must be 0..1')
    traces=readcsv(a.trace);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    artifacts,custody,truth=build(traces,a.seed,a.drop_rate,a.tamper_rate)
    predictions,metrics=audit(artifacts,truth)
    writecsv(out/'evidence_artifacts.csv',artifacts,['event_id','policy','episode','step','payload_json','sealed_sha256','stored_sha256'])
    writecsv(out/'custody_log.csv',custody,['event_id','operation','record_sha256'])
    writecsv(out/'private_ground_truth.csv',truth,['event_id','predecessor_event_id','policy','episode'])
    writecsv(out/'predicted_transitions.csv',predictions,['parent_event_id','child_event_id'])
    (out/'audit.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('E5-C20 | SYNTHETIC EVIDENCE AUDIT');print(json.dumps(metrics,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
