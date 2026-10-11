import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "causal"
OUT = ROOT / "results" / "causal"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
CAMPAIGNS = 100
rng = random.Random(SEED)

HOSTS = [f"host{i:02d}" for i in range(1, 21)]
STAGES = [
    "reconnaissance",
    "initial_access",
    "execution",
    "privilege_escalation",
    "credential_access",
    "lateral_movement",
    "objective",
]
SENSORS = ["network", "endpoint", "authentication", "application"]

telemetry = []
truth = []
chains = []

def add_event(campaign, timestamp, src, dst, sensor,
              event_type, signal, malicious, stage=0,
              predecessor=""):
    eid = f"C{campaign:03d}_E{len(telemetry):07d}"

    telemetry.append({
        "event_id": eid,
        "campaign_id": campaign,
        "timestamp": timestamp,
        "source_host": src,
        "destination_host": dst,
        "sensor": sensor,
        "event_type": event_type,
        "signal": round(signal, 5),
    })

    truth.append({
        "event_id": eid,
        "campaign_id": campaign,
        "malicious": int(malicious),
        "stage": stage,
        "predecessor_event_id": predecessor,
    })
    return eid

for campaign in range(CAMPAIGNS):
    base = campaign * 10000
    path = rng.sample(HOSTS, 5)
    attack_ids = []

    current_time = base + 100
    observable_types = [
        "connection", "login", "process_start",
        "file_access", "configuration_change",
        "service_request",
    ]

    for stage_idx, stage in enumerate(STAGES, start=1):
        current_time += rng.randint(12, 20)
        timestamp = current_time

        if stage_idx <= 2:
            src, dst = path[0], path[1]
        elif stage_idx <= 4:
            src, dst = path[1], path[2]
        elif stage_idx <= 6:
            src, dst = path[2], path[3]
        else:
            src, dst = path[3], path[4]

        predecessor = attack_ids[-1] if attack_ids else ""

        eid = add_event(
            campaign, timestamp, src, dst,
            rng.choice(SENSORS),
            rng.choice(observable_types),
            min(1, max(0, rng.gauss(0.72, 0.18))),
            True, stage_idx, predecessor
        )
        attack_ids.append(eid)

    chains.append({
        "campaign_id": campaign,
        "attack_event_ids": attack_ids,
        "attack_hosts": path,
    })

    # Benign activity uses the SAME host population.
    for _ in range(100):
        src = rng.choice(HOSTS)
        dst = rng.choice(HOSTS)
        timestamp = base + rng.randint(50, 400)

        add_event(
            campaign, timestamp, src, dst,
            rng.choice(SENSORS),
            rng.choice([
                "connection",
                "login",
                "process_start",
                "file_access",
                "configuration_change",
                "service_request",
            ]),
            min(1, max(0, rng.gauss(0.46, 0.22))),
            False
        )

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv(DATA / "telemetry.csv", telemetry)
write_csv(DATA / "ground_truth.csv", truth)

with (DATA / "chains_ground_truth.json").open("w", encoding="utf-8") as f:
    json.dump(chains, f, indent=2)

forbidden = {"malicious", "stage", "kind", "predecessor_event_id"}
assert not forbidden.intersection(telemetry[0])

# Verify ground-truth chain references.
truth_by_id = {r["event_id"]: r for r in truth}
for chain in chains:
    ids = chain["attack_event_ids"]
    assert len(ids) == 7
    for i in range(1, len(ids)):
        assert truth_by_id[ids[i]]["predecessor_event_id"] == ids[i - 1]

summary = {
    "seed": SEED,
    "campaigns": CAMPAIGNS,
    "events": len(telemetry),
    "attack_events": sum(r["malicious"] for r in truth),
    "benign_events": sum(not r["malicious"] for r in truth),
    "attack_stages_per_campaign": len(STAGES),
    "causal_chains_verified": len(chains),
    "forbidden_fields_exposed": [],
    "status": "PASS",
}

with (OUT / "e5c1_audit.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=" * 65)
print("E5-C1 | CAUSAL ATTACK CAMPAIGN GENERATION")
print("=" * 65)
for key, value in summary.items():
    print(f"{key:28s}: {value}")
print("=" * 65)
