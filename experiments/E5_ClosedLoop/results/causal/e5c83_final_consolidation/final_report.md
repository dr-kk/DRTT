# E5-C83 — Final Consolidated Experimental Report

**Scope:** E5-C79 through E5-C82 only. No new experiment, model fitting, or test-time selection.
**Data:** Five independent synthetic generator seeds per experiment; different seed groups between C79/C80/C81.
**Budgets:** 1,800 alerts and 400 predicted edges per seed in the reported tests.

## E5-C79: Structure-aware alert selection

| Telemetry | ScoreOnly F1 | TopologyStrong F1 | ScoreOnly edge coverage | TopologyStrong edge coverage |
| --- | --- | --- | --- | --- |
| original | 0.1404 | 0.1404 | 0.2710 | 0.2710 |
| matched_signal | 0.0260 | 0.0260 | 0.0497 | 0.0497 |

**Interpretation:** The validation-selected structure-aware policy does not improve held-out F1 or true-edge coverage over score-only selection.

## E5-C80: Inference-time feature-masking sensitivity

| Masked component | OriginalOnly / original F1 | OriginalOnly / matched F1 |
| --- | --- | --- |
| none | 0.1356 | 0.0268 |
| temporal | 0.1120 | 0.0228 |
| topology | 0.0184 | 0.0008 |
| signal | 0.1344 | 0.0296 |
| all | 0.0128 | 0.0032 |

**Interpretation:** The frozen model is highly sensitive to topology masking. These are inference-time zero-masking tests, not independently retrained ablations; correlated and derived features may remain.

## E5-C81: Equal-budget heuristic baseline comparison

| Method | Original F1 | Matched-signal F1 |
| --- | --- | --- |
| Frozen | 0.1268 | 0.0244 |
| HardNegative | 0.1276 | 0.0244 |
| RecentParent | 0.0180 | 0.0028 |
| HostHandoff | 0.0264 | 0.0016 |
| TemporalHost | 0.0556 | 0.0052 |
| SeededRandom | 0.0132 | 0.0012 |

**Interpretation:** Learned models outperform these tested heuristic baselines numerically. These baselines are not published state-of-the-art systems.

## E5-C82: Paired inference

| Telemetry | Comparison | F1 delta | Paired bootstrap 95% CI | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- |
| original | HardNegative vs Frozen | +0.0008 | [-0.0036, +0.0052] | 0.8750 | 1.0000 |
| original | HardNegative vs TemporalHost | +0.0720 | [+0.0580, +0.0888] | 0.0625 | 0.5000 |
| original | Frozen vs TemporalHost | +0.0712 | [+0.0584, +0.0844] | 0.0625 | 0.5000 |
| original | HardNegative vs SeededRandom | +0.1144 | [+0.1056, +0.1228] | 0.0625 | 0.5000 |
| matched_signal | HardNegative vs Frozen | +0.0000 | [-0.0020, +0.0016] | 1.0000 | 1.0000 |
| matched_signal | HardNegative vs TemporalHost | +0.0192 | [+0.0152, +0.0232] | 0.0625 | 0.5000 |
| matched_signal | Frozen vs TemporalHost | +0.0192 | [+0.0164, +0.0216] | 0.0625 | 0.5000 |
| matched_signal | HardNegative vs SeededRandom | +0.0232 | [+0.0204, +0.0260] | 0.0625 | 0.5000 |

**Inference:** No comparison reaches Holm-adjusted p < 0.05. Bootstrap intervals with only five seed clusters are exploratory; a percentile bootstrap interval excluding zero does not override the non-significant exact test.

## Limitations and claim boundaries

1. All results are synthetic; external dataset and deployment generalization are not established.
2. Matched-signal perturbations use benign labels to construct the evaluation distribution; this is an evaluation oracle, not an operational procedure.
3. Alert selection strongly limits true-edge coverage; topology-aware reranking cannot recover omitted edges.
4. Original and matched-signal scores cannot be described as robustly equivalent.
5. Five paired seeds limit inferential resolution; the smallest two-sided exact sign-flip p with five nonzero pairs is 0.0625.
6. C79, C80 and C81 use different independent test seeds. Compare methods within an experiment, not across experiments as if paired.
7. Inference-time masking is not a retrained feature-removal ablation.
8. Heuristic comparisons do not substantiate superiority over published SOTA methods.

## Suggested manuscript-safe conclusion

Under equal alert and edge budgets on independent synthetic seeds, the learned reconstruction models achieved higher F1 than the tested heuristic baselines, while hard-negative retraining offered only a small and statistically inconclusive gain over the frozen model. Inference-time masking indicates strong sensitivity to topology-related features. However, structure-aware alert selection did not improve held-out reconstruction, and matched-signal perturbations caused severe performance degradation. These findings identify alert coverage and distribution shift as unresolved limitations rather than solved capabilities.
