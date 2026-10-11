import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

def load(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def auc(pos,neg):
    # Exact Mann-Whitney AUC with midranks for ties.
    x=sorted([(v,1) for v in pos]+[(v,0) for v in neg])
    n1=len(pos);n0=len(neg)
    if not n1 or not n0:return None
    rank_sum=0.;i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[j][0]==x[i][0]:j+=1
        rank=(i+1+j)/2
        rank_sum+=rank*sum(y for _,y in x[i:j]);i=j
    return (rank_sum-n1*(n1+1)/2)/(n1*n0)

def main():
    ap=argparse.ArgumentParser(description='E5-C35 development-only observable context feasibility')
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--window',type=int,default=90)
    a=ap.parse_args();d=Path(a.dataset);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    events=load(d/'telemetry.csv');truth=load(d/'ground_truth.csv')
    expected={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    if not events or set(events[0])!=expected:raise ValueError('Unexpected telemetry schema')
    labels={r['event_id']:int(r['malicious']) for r in truth}
    if len(labels)!=len(events) or set(labels)!={r['event_id'] for r in events}:raise ValueError('Truth alignment failed')
    events.sort(key=lambda e:(int(e['timestamp']),e['event_id']))
    history=defaultdict(list);pair_history=defaultdict(list);rows=[]
    # STRICTLY PRIOR timestamps; same-timestamp events are processed as one batch.
    i=0
    while i<len(events):
        j=i+1;t=int(events[i]['timestamp'])
        while j<len(events) and int(events[j]['timestamp'])==t:j+=1
        for e in events[i:j]:
            src=e['source_host'];dst=e['destination_host']
            hs=history[src];hd=history[dst];hp=pair_history[(src,dst)]
            recent_src=sum(t-z<=a.window for z in hs)
            recent_dst=sum(t-z<=a.window for z in hd)
            recent_pair=sum(t-z<=a.window for z in hp)
            prev_src=(t-hs[-1]) if hs else a.window+1
            prev_dst=(t-hd[-1]) if hd else a.window+1
            rows.append({'event_id':e['event_id'],'malicious':labels[e['event_id']],
                         'signal':float(e['signal']),
                         'prior_src_activity':recent_src,
                         'prior_dst_activity':recent_dst,
                         'prior_pair_activity':recent_pair,
                         'prior_src_recency':1/(1+prev_src),
                         'prior_dst_recency':1/(1+prev_dst)})
        for e in events[i:j]:
            src=e['source_host'];dst=e['destination_host']
            history[src].append(t);history[dst].append(t);pair_history[(src,dst)].append(t)
        i=j
    features=['signal','prior_src_activity','prior_dst_activity','prior_pair_activity','prior_src_recency','prior_dst_recency']
    metrics=[]
    for feat in features:
        pos=[r[feat] for r in rows if r['malicious']==1];neg=[r[feat] for r in rows if r['malicious']==0]
        score=auc(pos,neg)
        metrics.append({'feature':feat,'malicious_mean':sum(pos)/len(pos),'benign_mean':sum(neg)/len(neg),
                        'auc':score,'orientation_free_auc':max(score,1-score)})
    with (out/'feature_feasibility.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    report={'experiment':'E5-C35','dataset':str(d),'events':len(events),'malicious_events':sum(labels.values()),
            'window':a.window,'causal_feature_policy':'strictly prior timestamps; same timestamp excluded',
            'label_usage':'labels used only for post-hoc AUC, never feature construction',
            'feature_metrics':metrics,'limitations':'univariate development-set feasibility only; no model trained; no test-set claim'}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C35 | CONTEXT FEASIBILITY (DEVELOPMENT ONLY)')
    for m in metrics:print(f"{m['feature']:>22}: AUC={m['auc']:.4f} | malicious mean={m['malicious_mean']:.4f} | benign mean={m['benign_mean']:.4f}")
    print('Saved:',out.resolve())
if __name__=='__main__':main()
