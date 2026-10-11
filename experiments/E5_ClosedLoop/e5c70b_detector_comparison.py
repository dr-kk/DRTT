import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path("src/causal").resolve()))

import e5c5_overlapping_generator as gen
import e5c38_end_to_end as det

train = Path("data/causal/e5c15_train")

def load_csv(path):
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

train_events = load_csv(train / "telemetry.csv")
train_truth = {
    r["event_id"]: int(r["malicious"])
    for r in load_csv(train / "ground_truth.csv")
}

def matrix(events, truth):
    features = det.extract(events, 90)
    X = np.asarray(
        [features[e["event_id"]] for e in events],
        dtype=float
    )
    y = np.asarray(
        [truth[e["event_id"]] for e in events],
        dtype=int
    )
    return X, y

X_train, y_train = matrix(train_events, train_truth)

models = {}

for name, columns in {
    "signal_only": [0],
    "full_four_features": [0, 1, 2, 3],
}.items():
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0)
    )
    model.fit(X_train[:, columns], y_train)
    models[name] = (model, columns)

print("E5-C70B | Held-out detector comparison")
print("Training events:", len(y_train))
print()

for seed in [7001, 7002, 7003, 7004, 7005]:
    events, truth_rows = gen.generate(
        seed=seed,
        n_campaigns=100,
        duration=2400,
        background_events=10000
    )

    truth = {
        r["event_id"]: int(r["malicious"])
        for r in truth_rows
    }

    X, y = matrix(events, truth)

    print(f"Seed {seed} | Prevalence={y.mean():.4f}")

    for name, (model, columns) in models.items():
        probabilities = model.predict_proba(
            X[:, columns]
        )[:, 1]

        print(
            f"  {name:20s}"
            f" AUROC={roc_auc_score(y, probabilities):.4f}"
            f" AP={average_precision_score(y, probabilities):.4f}"
        )

    print(f"  random_ranking       AP={y.mean():.4f}")
    print()
