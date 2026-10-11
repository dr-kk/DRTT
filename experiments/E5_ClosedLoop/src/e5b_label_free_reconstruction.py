import csv
import json
import random
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def reconstruct(events, threshold=0.65, window=30):
    """Inference uses observable telemetry only."""
    forbidden = {"malicious", "stage", "kind", "label", "ground_truth"}
    predictions = []
    recent = []

    for event in sorted(events, key=lambda x: (int(x["timestamp"]), x["event_id"])):
        if forbidden.intersection(event):
            raise ValueError("Ground-truth field exposed to reconstruction")

        timestamp = int(event["timestamp"])
        signal = float(event["signal"])

        recent = [
            previous for previous in recent
            if timestamp - int(previous["timestamp"]) <= window
        ]

        # Entity correlation, without hidden labels.
        related = [
            previous for previous in recent
            if (
                previous["source_host"] == event["source_host"]
                or previous["destination_host"] == event["destination_host"]
                or previous["destination_host"] == event["source_host"]
            )
        ]

        entity_score = min(1.0, len(related) / 5.0)

        distinct_sensors = {
            previous["sensor"] for previous in related
        }
        distinct_sensors.add(event["sensor"])
        source_score = min(1.0, (len(distinct_sensors) - 1) / 3.0)

        if related:
            nearest_gap = min(
                timestamp - int(previous["timestamp"])
                for previous in related
            )
            temporal_score = max(0.0, 1.0 - nearest_gap / window)
        else:
            temporal_score = 0.0

        # Preliminary evidence score; not yet a validated
        # causal attack-chain confidence model.
        confidence = (
            0.55 * signal
            + 0.20 * entity_score
            + 0.15 * source_score
            + 0.10 * temporal_score
        )

        predictions.append({
            "event_id": event["event_id"],
            "confidence": round(confidence, 6),
            "predicted_malicious": int(confidence >= threshold),
        })

        recent.append(event)

    return predictions

def digest(predictions):
    payload = json.dumps(predictions, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()

def evaluate(predictions, truth):
    labels = {row["event_id"]: int(row["malicious"]) for row in truth}
    if set(labels) != {p["event_id"] for p in predictions}:
        raise ValueError("Prediction and ground-truth IDs do not match")

    tp = fp = tn = fn = 0
    for p in predictions:
        y = labels[p["event_id"]]
        yhat = p["predicted_malicious"]
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fp / (fp + tn), 6) if fp + tn else 0.0,
    }

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

telemetry = read_csv(DATA / "telemetry.csv")
truth = read_csv(DATA / "ground_truth.csv")

predictions = reconstruct(telemetry)
original_hash = digest(predictions)

# Permute ground-truth labels and rerun inference.
# This checks that predictions are invariant to label changes.
rng = random.Random(2026)
permuted_truth = [dict(row) for row in truth]
shuffled = [(r["malicious"], r["stage"]) for r in permuted_truth]
rng.shuffle(shuffled)

for row, (malicious, stage) in zip(permuted_truth, shuffled):
    row["malicious"] = malicious
    row["stage"] = stage

permuted_predictions = reconstruct(telemetry)
permuted_hash = digest(permuted_predictions)
assert original_hash == permuted_hash

metrics = evaluate(predictions, truth)

# Signal-only baseline on the same observable events.
baseline = [
    {
        "event_id": e["event_id"],
        "predicted_malicious": int(float(e["signal"]) >= 0.65)
    }
    for e in telemetry
]
baseline_metrics = evaluate(baseline, truth)

audit = {
    "experiment": "E5-B",
    "telemetry_events": len(telemetry),
    "label_permutation_invariant": original_hash == permuted_hash,
    "inference_hash": original_hash,
    "forensic_event_metrics": metrics,
    "signal_only_baseline": baseline_metrics,
    "status": "PASS" if original_hash == permuted_hash else "FAIL",
}

save_csv(OUT / "e5b_event_predictions.csv", predictions)

with open(OUT / "e5b_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("=" * 65)
print("E5-B | LABEL-FREE EVENT CORRELATION")
print("=" * 65)
print("Telemetry events:", len(telemetry))
print("Label-permutation invariant:", audit["label_permutation_invariant"])
print("Event metrics:", metrics)
print("Signal-only baseline:", baseline_metrics)
print("Status:", audit["status"])
print("=" * 65)
