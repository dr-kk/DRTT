"""E5-C12 corrected consistency audit. Ground truth is used only for scoring."""
import argparse
import csv
import json
import sys
from pathlib import Path

def read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def evaluate_rankings(rankings, gold):
    predicted={(opts[0],child) for child,opts in rankings.items() if opts}
    return {'predicted_edges':len(predicted), 'true_positives':len(predicted & gold),
            'top_k_hits':{str(k):sum(parent in rankings.get(child,[])[:k] for parent,child in gold)
                          for k in (1,3,5,10)}}

def graph_score(parent, child, window=90):
    gap=int(child['timestamp'])-int(parent['timestamp'])
    continuity=parent['destination_host']==child['source_host']
    same_src=parent['source_host']==child['source_host']
    same_dst=parent['destination_host']==child['destination_host']
    return ((.6 if continuity else 0)+(.25 if same_src or same_dst else 0)
            +.15*(1-gap/window))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--telemetry', type=Path, required=True)
    ap.add_argument('--truth', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import e5c9_ranking_audit as c9
    import e5c11_temporal_host_baseline as c11
    rows=read(args.telemetry)
    byid={r['event_id']:r for r in rows}
    selected,r9=c9.rank(rows)
    r11=c11.infer(rows)['frozen_graph']
    if set(r9)!=set(r11):
        raise RuntimeError('Ranked child sets differ')
    differences=[]
    same_candidates=0
    different_candidates=0
    top1_different=0
    for child in sorted(r9):
        x,y=r9[child],r11[child]
        if x==y:
            continue
        same=set(x)==set(y)
        same_candidates+=int(same)
        different_candidates+=int(not same)
        top1_different+=int((x[0] if x else None)!=(y[0] if y else None))
        differences.append({'child':child,'c9_top1':x[0] if x else '',
                            'c11_top1':y[0] if y else '',
                            'same_candidate_set':same})
    gold={(r['predecessor_event_id'],r['event_id']) for r in read(args.truth)
          if r['predecessor_event_id']}
    tied_best=0
    tied_top1_correct=0
    for child, options in r11.items():
        if len(options)<2:
            continue
        scores=[graph_score(byid[parent_id],byid[child]) for parent_id in options]
        # Floating point equality mirrors the sorting keys used in the original scripts.
        best=max(scores)
        if sum(s==best for s in scores)>1:
            tied_best+=1
            if (options[0],child) in gold:
                tied_top1_correct+=1
    report={'dataset':str(args.telemetry),'true_edges':len(gold),
            'c9':evaluate_rankings(r9,gold),
            'c11_frozen_graph':evaluate_rankings(r11,gold),
            'children_with_different_rankings':len(differences),
            'children_with_different_top1':top1_different,
            'different_rankings_same_candidate_set':same_candidates,
            'different_candidate_sets':different_candidates,
            'children_with_tied_best_score':tied_best,
            'correct_c11_top1_with_tied_best_score':tied_top1_correct,
            'note':'Exploratory diagnostic; seed 2027 has already been inspected. Scores and candidate IDs are computed from telemetry only.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'consistency_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (args.output/'ranking_differences.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['child','c9_top1','c11_top1','same_candidate_set'])
        w.writeheader()
        w.writerows(differences)
    print('E5-C12 | CORRECTED RANKING CONSISTENCY AUDIT')
    print(json.dumps(report,indent=2))
    print('Saved:',args.output.resolve())

if __name__=='__main__':
    main()
