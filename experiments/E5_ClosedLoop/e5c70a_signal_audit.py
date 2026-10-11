import csv
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

root = Path("data/causal/e5c15_train")

with (root / "telemetry.csv").open(newline="", encoding="utf-8") as f:
    telemetry = list(csv.DictReader(f))

with (root / "ground_truth.csv").open(newline="", encoding="utf-8") as f:
    truth = {
        r["event_id"]: int(r["malicious"])
        for r in csv.DictReader(f)
    }

y = [truth[r["event_id"]] for r in telemetry]
signal = [float(r["signal"]) for r in telemetry]

print("Total events:", len(y))
print("Malicious events:", sum(y))
print("Malicious prevalence:", round(sum(y) / len(y), 6))
print("Signal-only AUROC:", round(roc_auc_score(y, signal), 6))
print("Signal-only average precision:", round(
    average_precision_score(y, signal), 6
))
