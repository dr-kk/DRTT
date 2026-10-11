import argparse, csv, json, random
from pathlib import Path

EVENT_TYPES = ['connection','login','process_start','file_access','configuration_change','service_request']
SENSORS = ['network','endpoint','authentication','application']
HOSTS = [f'host{i:02d}' for i in range(1,21)]


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def generate(seed, n_campaigns, duration, background_events):
    rng=random.Random(seed)
    raw=[]
    # The hidden campaign identifier and attack stage exist only in private truth.
    for c in range(n_campaigns):
        start=rng.randint(100, duration-400)
        hosts=rng.sample(HOSTS, 5)
        t=start
        parent=None
        for stage in range(1,8):
            t+=rng.randint(8,25)
            src,dst=([hosts[0],hosts[1]] if stage<=2 else
                     [hosts[1],hosts[2]] if stage<=4 else
                     [hosts[2],hosts[3]] if stage<=6 else
                     [hosts[3],hosts[4]])
            internal=f'attack_{c}_{stage}'
            raw.append(dict(internal=internal,t=t,src=src,dst=dst,
                            sensor=rng.choice(SENSORS),etype=rng.choice(EVENT_TYPES),
                            signal=round(max(0,min(1,rng.gauss(.68,.22))),5),
                            malicious=1,stage=stage,campaign=c,parent=parent))
            parent=internal
    for b in range(background_events):
        raw.append(dict(internal=f'benign_{b}',t=rng.randint(0,duration),
                        src=rng.choice(HOSTS),dst=rng.choice(HOSTS),
                        sensor=rng.choice(SENSORS),etype=rng.choice(EVENT_TYPES),
                        signal=round(max(0,min(1,rng.gauss(.47,.22))),5),
                        malicious=0,stage=0,campaign='',parent=None))
    # Sort chronologically, then replace internal IDs with opaque sequential IDs.
    # Note: monotone IDs are observable timestamps in disguise, not campaign identifiers.
    raw.sort(key=lambda r:(r['t'],rng.random()))
    idmap={r['internal']:f'EV{i:08d}' for i,r in enumerate(raw)}
    telemetry=[]; truth=[]
    for r in raw:
        eid=idmap[r['internal']]
        telemetry.append(dict(event_id=eid,timestamp=r['t'],source_host=r['src'],
                              destination_host=r['dst'],sensor=r['sensor'],
                              event_type=r['etype'],signal=r['signal']))
        truth.append(dict(event_id=eid,malicious=r['malicious'],stage=r['stage'],
                          campaign_id=r['campaign'],
                          predecessor_event_id=idmap[r['parent']] if r['parent'] else ''))
    return telemetry,truth


def audit(telemetry,truth,n_campaigns):
    assert len({r['event_id'] for r in telemetry})==len(telemetry)
    assert set(telemetry[0])=={'event_id','timestamp','source_host','destination_host','sensor','event_type','signal'}
    byid={r['event_id']:r for r in telemetry}
    groups={}
    for r in truth:
        if r['malicious']:
            groups.setdefault(r['campaign_id'],[]).append(r)
            if r['predecessor_event_id']:
                assert int(byid[r['predecessor_event_id']]['timestamp']) < int(byid[r['event_id']]['timestamp'])
    assert len(groups)==n_campaigns and all(len(g)==7 for g in groups.values())
    intervals=[(min(int(byid[r['event_id']]['timestamp']) for r in g),
                max(int(byid[r['event_id']]['timestamp']) for r in g)) for g in groups.values()]
    overlapping_pairs=sum(a[0]<=b[1] and b[0]<=a[1] for i,a in enumerate(intervals) for b in intervals[i+1:])
    timestamps=sorted(int(r['timestamp']) for r in telemetry)
    gaps=[b-a for a,b in zip(timestamps,timestamps[1:])]
    return {'events':len(telemetry),'malicious_events':sum(r['malicious'] for r in truth),
            'true_edges':sum(bool(r['predecessor_event_id']) for r in truth),
            'overlapping_campaign_pairs':overlapping_pairs,
            'maximum_timestamp_gap':max(gaps,default=0),
            'observable_columns':list(telemetry[0]),
            'note':'Synthetic correlated attacks; not executable causal response or independent real-world validation'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/causal/e5c5')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--campaigns',type=int,default=100)
    p.add_argument('--duration',type=int,default=2400)
    p.add_argument('--background',type=int,default=10000)
    a=p.parse_args()
    if a.duration<=400 or a.campaigns<2 or a.background<1:
        p.error('duration must exceed 400, campaigns >= 2, background >= 1')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    telemetry,truth=generate(a.seed,a.campaigns,a.duration,a.background)
    report=audit(telemetry,truth,a.campaigns)
    write_csv(out/'telemetry.csv',telemetry,list(telemetry[0]))
    write_csv(out/'ground_truth.csv',truth,list(truth[0]))
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('E5-C5 | OVERLAPPING CAMPAIGN GENERATION')
    print(json.dumps(report,indent=2))
    print('Saved:',out.resolve())
    if report['overlapping_campaign_pairs']==0:
        raise RuntimeError('No campaign overlap observed; change parameters')

if __name__=='__main__':main()
