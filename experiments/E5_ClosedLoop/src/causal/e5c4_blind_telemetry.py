import csv, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'causal'; OUT=ROOT/'results'/'causal'; OUT.mkdir(parents=True,exist_ok=True)
def load(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
events=load(DATA/'telemetry.csv'); truth=load(DATA/'ground_truth.csv')
assert len(events)==len(truth) and len({r['event_id'] for r in events})==len(events)
# Deterministic opaque IDs remove campaign-index and row-index strings.
ids={r['event_id']:'ev_'+hashlib.sha256(('e5c4-blind-id-v1|'+r['event_id']).encode()).hexdigest()[:20] for r in events}
assert len(set(ids.values()))==len(ids)
observable=['event_id','timestamp','source_host','destination_host','sensor','event_type','signal']
blind=[{k:(ids[r['event_id']] if k=='event_id' else r[k]) for k in observable} for r in events]
blind.sort(key=lambda r:(int(r['timestamp']),r['event_id']))
blind_truth=[]
for r in truth:
 blind_truth.append({'event_id':ids[r['event_id']],'malicious':r['malicious'],'stage':r['stage'],'predecessor_event_id':ids[r['predecessor_event_id']] if r['predecessor_event_id'] else ''})
write(DATA/'telemetry_blind.csv',blind)
write(DATA/'ground_truth_blind.csv',blind_truth)
# Audit known synthetic artifacts: timing gaps remain, even after dropping campaign_id.
times=sorted(int(r['timestamp']) for r in blind)
gaps=[b-a for a,b in zip(times,times[1:])]
large=sum(g>1000 for g in gaps)
report={'events':len(blind),'campaign_id_exposed':False,'event_id_prefix_exposed':False,'large_time_gaps_over_1000':large,'maximum_time_gap':max(gaps),'note':'Blind IDs and no campaign_id, but synthetic time gaps still leak segmentation. Not publication-ready.'}
(OUT/'e5c4_blind_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('E5-C4 | BLIND TELEMETRY AUDIT')
for k,v in report.items():print(f'{k}: {v}')
print('Saved:',DATA/'telemetry_blind.csv')
print('Saved:',DATA/'ground_truth_blind.csv')
