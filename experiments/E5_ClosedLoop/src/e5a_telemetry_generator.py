import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_EVENTS = 5000
rng = random.Random(SEED)

HOSTS = [f"host{i:02d}" for i in range(1, 21)]
SENSORS = ["network", "endpoint", "authentication", "application"]
EVENT_TYPES = [
    "connection",
    "process_start",
    "login",
    "file_access",
    "service_request",
    "configuration_change",
]

telemetry = []
ground_truth = []

for i in range(N_EVENTS):
    event_id = f"E{i:06d}"
    timestamp = 1_700_000_000 + i
    source = rng.choice(HOSTS)
    destination = rng.choice(HOSTS)
    sensor = rng.choice(SENSORS)
    event_type = rng.choice(EVENT_TYPES)

    # Hidden labels used only by evaluation.
    malicious = int(rng.random() < 0.15)
    stage = rng.randint(1, 7) if malicious else 0

    # Observable features with deliberately overlapping distributions.
    signal = min(1.0, max(0.0, rng.gauss(
        0.62 if malicious else 0.43, 0.22
    )))

    telemetry.append({
        "event_id": event_id,
        "timestamp": timestamp,
        "source_host": source,
        "destination_host": destination,
        "sensor": sensor,
        "event_type": event_type,
        "signal": round(signal, 5),
    })

    ground_truth.append({
        "event_id": event_id,
        "malicious": malicious,
        "stage": stage,
    })

def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save_csv(DATA / "telemetry.csv", telemetry)
save_csv(DATA / "ground_truth.csv", ground_truth)

telemetry_fields = set(telemetry[0])
forbidden = {"malicious", "stage", "kind", "label", "ground_truth"}

assert not telemetry_fields.intersection(forbidden)

attack_hosts = {
    r["source_host"] for r, gt in zip(telemetry, ground_truth)
    if gt["malicious"]
}
benign_hosts = {
    r["source_host"] for r, gt in zip(telemetry, ground_truth)
    if not gt["malicious"]
}

overlap = attack_hosts.intersection(benign_hosts)
assert len(overlap) > 0

summary = {
    "seed": SEED,
    "total_events": N_EVENTS,
    "malicious_events": sum(r["malicious"] for r in ground_truth),
    "benign_events": sum(1 - r["malicious"] for r in ground_truth),
    "host_overlap": len(overlap),
    "forbidden_fields_exposed": sorted(
        telemetry_fields.intersection(forbidden)
    ),
    "status": "PASS",
}

with (RESULTS / "e5a_audit.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=" * 65)
print("E5-A | LABEL-ISOLATED TELEMETRY GENERATION")
print("=" * 65)
for key, value in summary.items():
    print(f"{key:26s}: {value}")
print("=" * 65)
print("Saved telemetry.csv, ground_truth.csv, e5a_audit.json")
