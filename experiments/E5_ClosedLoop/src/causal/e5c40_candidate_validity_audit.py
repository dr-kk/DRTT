"""E5-C40: descriptive candidate-validity and frozen-gate audit of E5-C39.
No training, threshold search, or model modifications. Uses existing diagnostic CSV.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

BINS = [(0,.1),(.1,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0000001)]

def load(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    needed={'seed','pipeline','child','parent','true_parent','true_parent_selected',
            'true_parent_candidate','top1_correct','candidate_count','gate_probability','accepted'}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError('Missing expected E5-C39 candidate diagnostic columns')
    for r in rows:
        for k in ('true_parent_selected','true_parent_candidate','top1_correct','accepted'):
            r[k]=int(r[k])
            if r[k] not in (0,1):raise ValueError(f'Invalid binary {k}')
        r['candidate_count']=int(r['candidate_count'])
        r['gate_probability']=float(r['gate_probability'])
        if not 0<=r['gate_probability']<=1:raise ValueError('Invalid probability')
        if r['top1_correct'] and not r['true_parent_candidate']:
            raise ValueError('Correct top1 without true candidate')
        r['validity']='no_true_parent_candidate' if not r['true_parent_candidate'] else ('top1_correct' if r['top1_correct'] else 'wrong_top1_with_true_candidate')
    keys=[(r['seed'],r['pipeline'],r['child']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate child rows')
    return rows

def stat(rows):
    n=len(rows); acc=[r for r in rows if r['accepted']]; correct=[r for r in rows if r['top1_correct']]
    return {'candidate_children':n,'true_parent_available':sum(r['true_parent_candidate'] for r in rows),
            'top1_correct':len(correct),'accepted':len(acc),'accepted_tp':sum(r['top1_correct'] for r in acc),
            'accepted_fp':sum(not r['top1_correct'] for r in acc),
            'gate_prob_mean':mean(r['gate_probability'] for r in rows) if n else None,
            'gate_prob_mean_accepted':mean(r['gate_probability'] for r in acc) if acc else None,
            'gate_prob_mean_correct_top1':mean(r['gate_probability'] for r in correct) if correct else None,
            'gate_prob_mean_wrong_top1':mean(r['gate_probability'] for r in rows if not r['top1_correct']) if len(correct)<n else None,
            'accepted_precision':sum(r['top1_correct'] for r in acc)/len(acc) if acc else None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=load(a.input)
    pipelines=sorted(set(r['pipeline'] for r in rows))
    if set(pipelines)!={'signal','context'}:raise ValueError('Expected signal and context pipelines')
    a.output.mkdir(parents=True,exist_ok=True)
    per=[];bins=[];groups=defaultdict(list)
    for r in rows:groups[(r['seed'],r['pipeline'])].append(r)
    for (seed,pipeline),rr in sorted(groups.items()):per.append({'seed':seed,'pipeline':pipeline,**stat(rr)})
    overall={}
    for pipe in pipelines:
        rr=[r for r in rows if r['pipeline']==pipe]
        overall[pipe]=stat(rr)
        overall[pipe]['validity_groups']={v:stat([r for r in rr if r['validity']==v]) for v in ('no_true_parent_candidate','wrong_top1_with_true_candidate','top1_correct')}
        for low,high in BINS:
            part=[r for r in rr if low<=r['gate_probability']<high]
            accepted=[r for r in part if r['accepted']]
            bins.append({'pipeline':pipe,'probability_bin':f'[{low:.1f},{min(high,1):.1f}{"]" if high>1 else ")"}',
                         'candidate_children':len(part),'correct_top1':sum(r['top1_correct'] for r in part),
                         'empirical_correctness':sum(r['top1_correct'] for r in part)/len(part) if part else '',
                         'mean_gate_probability':mean(r['gate_probability'] for r in part) if part else '',
                         'accepted':len(accepted),'accepted_tp':sum(r['top1_correct'] for r in accepted)})
    seeds=sorted({r['seed'] for r in rows})
    paired=[]
    for seed in seeds:
        d={r['pipeline']:r for r in per if r['seed']==seed}
        if set(d)!=set(pipelines):raise ValueError('Unpaired seed '+seed)
        paired.append({'seed':seed,'delta_accepted_tp':d['context']['accepted_tp']-d['signal']['accepted_tp'],
                       'delta_accepted_fp':d['context']['accepted_fp']-d['signal']['accepted_fp'],
                       'delta_true_parent_available':d['context']['true_parent_available']-d['signal']['true_parent_available'],
                       'delta_top1_correct':d['context']['top1_correct']-d['signal']['top1_correct']})
    summary={'experiment':'E5-C40','scope':'Post-hoc descriptive audit of previously evaluated synthetic E5-C38/E5-C39 seeds; no tuning',
             'input':str(a.input),'seeds':seeds,'pipelines':overall,'paired_differences':paired,
             'definitions':{'true_parent_available':'True predecessor is in candidate set; stage-one and benign events have no true predecessor',
                            'top1_correct':'Frozen reranker selected exact true predecessor',
                            'accepted_tp':'Gate accepted exact true predecessor',
                            'gate_probability':'Model-estimated top1 correctness, not calibrated causal certainty'},
             'limitations':['Test labels used for diagnostics only','Bins are descriptive, not recalibration','Same synthetic generator across seeds','No causal or operational validity inferred']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def write(name,items):
        if not items:return
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    write('per_seed_audit.csv',per);write('probability_bins.csv',bins);write('paired_differences.csv',paired)
    print('E5-C40 | CANDIDATE VALIDITY AND GATE AUDIT (NO TUNING)')
    for pipe in ('signal','context'):
        s=overall[pipe]
        print(f"{pipe}: candidate={s['candidate_children']}, parent_available={s['true_parent_available']}, top1_correct={s['top1_correct']}, accepted TP/FP={s['accepted_tp']}/{s['accepted_fp']}")
        for k,v in s['validity_groups'].items():
            print(f"  {k}: n={v['candidate_children']}, accepted={v['accepted']}, mean_gate_prob={v['gate_prob_mean']:.4f}" if v['gate_prob_mean'] is not None else f'  {k}: empty')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
