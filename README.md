# DRTT

## Dynamic Red Teaming and Forensic Triangulation

Reproducibility artifacts for the manuscript:

**DRTT: Evidence-Grounded Causal Reconstruction Under Synthetic Network Distribution Shifts**

Prepared for IEEE Network.

## Experiments

- E1: PPO Conditional Dynamic-Graph Rerouting
- E2: Hard and Incomplete Telemetry Benchmark
- E3: DRTT Component Ablation
- E4: Robustness to Telemetry Loss

## Main Results

- E1: 30 paired seeds; conditional primary-to-alternate route switching in all evaluated pairs.
- E2: DRTT reduced mean IRT by 52.6% relative to Static Defense.
- E2: Detection improved by 22.86 percentage points.
- E2: Forensic reconstruction accuracy improved by 34.43 percentage points.
- E3: Ablation isolates adaptive response, forensic triangulation, and feedback contributions.
- E4: DRTT retained an IRT, detection, and reconstruction advantage through 40% telemetry dropout.

## Scope

Results are from controlled synthetic cyber-range experiments and should not be interpreted as production-network performance.

## Reproduction

Install dependencies:

    pip install -r requirements.txt

Run the experiment scripts from the repository root.

## Results

Frozen result summaries and run-level outputs used in the manuscript are provided under the results directory.


## E5: Closed-Loop Forensic Reconstruction

E5 includes synthetic telemetry generation, causal attack-chain
reconstruction, forensic feedback, and independent evaluation.

### Key Reproducibility Artifacts

| Experiment | Description |
|---|---|
| E5-C69 | Frozen independent validation |
| E5-C80 | Component ablation |
| E5-C81 | Fair baseline comparison |
| E5-C82 | Paired statistical analysis |
| E5-C83 | Final consolidation, results, and figures |
| E5-C84 | Label-permutation audit |
| E5-C85 | Host-overlap stress evaluation |

### Source Code

- experiments/E5_ClosedLoop/
- experiments/E5_ClosedLoop/src/causal/

### Final Results

- experiments/E5_ClosedLoop/results/causal/e5c83_final_consolidation/

All results originate from controlled synthetic experiments.
They should not be interpreted as production-network performance.

