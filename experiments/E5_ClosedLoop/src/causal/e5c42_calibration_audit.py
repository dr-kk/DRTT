#!/usr/bin/env python3
"""E5-C42: diagnostic calibration audit of existing E5-C41 predictions.
No fitting, threshold optimization, or evaluation on held-out test seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, precision_recall_fscore_support)


def safe_float(x):
    return None if not np.isfinite(x) else float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    required = {'child_event_id', 'true_parent_available', 'predicted_probability', 'predicted_available_at_0_5'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(df) == 0:
        raise ValueError('Input CSV is empty')
    if df['child_event_id'].duplicated().any():
        raise ValueError('Duplicate child_event_id values detected')
    y = pd.to_numeric(df['true_parent_available'], errors='raise').to_numpy()
    p = pd.to_numeric(df['predicted_probability'], errors='raise').to_numpy(dtype=float)
    existing = pd.to_numeric(df['predicted_available_at_0_5'], errors='raise').to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isin(existing, [0, 1]).all():
        raise ValueError('Binary label/prediction column contains non-binary values')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Predicted probabilities must be finite and in [0,1]')
    decision = (p >= .5).astype(int)
    if not np.array_equal(existing, decision):
        raise ValueError('Saved 0.5 decisions do not match probabilities')
    if len(np.unique(y)) < 2:
        raise ValueError('Both classes are required for AUROC/AP')

    args.output.mkdir(parents=True, exist_ok=True)
    n = len(y)
    base_rate = float(np.mean(y))
    brier = float(brier_score_loss(y, p))
    mean_p = float(np.mean(p))
    bins = np.linspace(0, 1, 11)
    # Include p=1.0 in final bin.
    bin_ids = np.minimum(np.searchsorted(bins, p, side='right') - 1, 9)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(10):
        mask = bin_ids == i
        count = int(mask.sum())
        confidence = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        gap = abs(confidence - observed) if count else None
        if count:
            ece += (count / n) * gap
            mce = max(mce, gap)
        rows.append({'bin_lower': round(float(bins[i]), 2),
                     'bin_upper': round(float(bins[i+1]), 2),
                     'count': count, 'positives': int(y[mask].sum()),
                     'mean_predicted_probability': confidence,
                     'observed_positive_fraction': observed,
                     'absolute_calibration_gap': gap})
    pd.DataFrame(rows).to_csv(args.output / 'reliability_bins.csv', index=False)

    pr, rc, f1, _ = precision_recall_fscore_support(y, decision, average='binary', zero_division=0)
    tp = int(((y == 1) & (decision == 1)).sum())
    fp = int(((y == 0) & (decision == 1)).sum())
    fn = int(((y == 1) & (decision == 0)).sum())
    tn = int(((y == 0) & (decision == 0)).sum())
    # These fixed bands describe behavior; they are NOT selected operating thresholds.
    bands = [(0.0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0)]
    band_rows = []
    for i, (lo, hi) in enumerate(bands):
        mask = (p >= lo) & ((p <= hi) if i == len(bands)-1 else (p < hi))
        band_rows.append({'probability_lower': lo, 'probability_upper': hi,
                          'count': int(mask.sum()), 'actual_available': int(y[mask].sum()),
                          'observed_rate': float(y[mask].mean()) if mask.any() else None,
                          'mean_probability': float(p[mask].mean()) if mask.any() else None})
    pd.DataFrame(band_rows).to_csv(args.output / 'probability_bands.csv', index=False)

    summary = {
        'experiment': 'E5-C42', 'scope': 'development calibration diagnostic only',
        'input': str(args.input), 'samples': n, 'positives': int(y.sum()),
        'base_rate': base_rate, 'mean_predicted_probability': mean_p,
        'calibration_in_the_large_error': mean_p-base_rate,
        'auroc': float(roc_auc_score(y,p)),
        'average_precision': float(average_precision_score(y,p)),
        'log_loss': float(log_loss(y,p,labels=[0,1])),
        'brier_score': brier, 'constant_base_rate_brier': base_rate*(1-base_rate),
        'brier_skill_vs_base_rate': 1-brier/(base_rate*(1-base_rate)),
        'ece_10_equal_width': float(ece), 'mce_10_equal_width': float(mce),
        'at_fixed_0_5': {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn,
                         'precision':float(pr),'recall':float(rc),'f1':float(f1)},
        'warning': 'This is a previously used development calibration set; metrics are descriptive and not independent validation. ECE is bin-dependent. No model was fitted or threshold optimized.'
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('E5-C42 | FROZEN CANDIDATE-VALIDITY CALIBRATION AUDIT')
    print(f'Samples={n}; positives={int(y.sum())}; prevalence={base_rate:.4%}')
    print(f'Mean predicted probability={mean_p:.4f}; observed prevalence={base_rate:.4f}')
    print(f'AUROC={summary["auroc"]:.4f}; AP={summary["average_precision"]:.4f}; log loss={summary["log_loss"]:.4f}')
    print(f'Brier={brier:.4f}; base-rate Brier={summary["constant_base_rate_brier"]:.4f}; Brier skill={summary["brier_skill_vs_base_rate"]:.4f}')
    print(f'ECE (10 equal-width bins)={ece:.4f}; MCE={mce:.4f}')
    print(f'Fixed 0.50 threshold: TP={tp} FP={fp} FN={fn} TN={tn}; precision={pr:.4f}; recall={rc:.4f}; F1={f1:.4f}')
    print('WARNING: development-only audit; no independent validation or threshold tuning.')
    print(f'Saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
