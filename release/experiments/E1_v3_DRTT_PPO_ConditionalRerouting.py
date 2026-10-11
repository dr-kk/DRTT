#!/usr/bin/env python
"""
E1_v3_DRTT_PPO_ConditionalRerouting.py

DRTT Experiment E1-v3:
PPO conditional rerouting under a defender-induced dynamic attack-graph change.

Key methodological change from E1-v2
------------------------------------
No direct reward is attached to the label PRIMARY or ALTERNATE.
Rewards represent:
  * attack progress,
  * objective completion,
  * action/step cost,
  * failed/redundant action cost,
  * detection exposure.

The primary route is operationally cheaper when open. The alternate route is
more costly but remains feasible after the defender removes the primary edge.

Paired evaluation uses the SAME trained PPO policy and the SAME seed in:
  A) PRIMARY_OPEN
  B) PRIMARY_BLOCKED

Outputs
-------
runs/e1_v3_conditional/
    model.zip
    paired_evaluation.csv
    paired_seed_summary.csv
    summary.json

Install
-------
python -m pip install gymnasium stable-baselines3 pandas numpy torch

Run
---
python E1_v3_DRTT_PPO_ConditionalRerouting.py
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


class DRTTEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode="TRAIN", seed=42, training_block_prob=0.5, max_steps=24):
        super().__init__()
        assert mode in {"TRAIN", "PRIMARY_OPEN", "PRIMARY_BLOCKED"}
        self.mode = mode
        self.training_block_prob = float(training_block_prob)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        # 0 recon-web, 1 exploit-web, 2 recon-app, 3 exploit-app,
        # 4 app->db, 5 recon-id, 6 credential-access, 7 id->db, 8 objective
        self.action_space = spaces.Discrete(9)

        # web_disc, web_comp, app_disc, app_comp, id_disc, credential, db_comp,
        # primary_edge, alternate_edge, intervention_seen, normalized_steps
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        self.reset(seed=seed)

    def _obs(self):
        return np.array([
            self.web_disc,
            self.web_comp,
            self.app_disc,
            self.app_comp,
            self.id_disc,
            self.credential,
            self.db_comp,
            self.primary_edge,
            self.alternate_edge,
            self.intervention_seen,
            min(self.steps / self.max_steps, 1.0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.web_disc = 0.0
        self.web_comp = 0.0
        self.app_disc = 0.0
        self.app_comp = 0.0
        self.id_disc = 0.0
        self.credential = 0.0
        self.db_comp = 0.0

        self.primary_edge = 1.0
        self.alternate_edge = 1.0
        self.intervention_seen = 0.0

        self.steps = 0
        self.invalid = 0
        self.route = "NONE"
        self.success = False
        self.intervention_decided = False

        return self._obs(), {}

    def _invalid(self):
        self.invalid += 1
        return -1.25

    def _decide_intervention(self):
        if self.intervention_decided:
            return

        if self.mode == "PRIMARY_BLOCKED":
            block = True
        elif self.mode == "PRIMARY_OPEN":
            block = False
        else:
            block = bool(self.rng.random() < self.training_block_prob)

        if block:
            self.primary_edge = 0.0
            self.intervention_seen = 1.0

        self.intervention_decided = True

    def step(self, action):
        self.steps += 1

        # Every interaction has a cost. Thus shorter feasible attack chains
        # are naturally preferable without assigning a route-specific bonus.
        reward = -0.15
        terminated = False
        truncated = False

        if action == 0:  # recon web
            if not self.web_disc:
                self.web_disc = 1.0
                reward += 0.50
            else:
                reward += self._invalid()

        elif action == 1:  # exploit web
            if self.web_disc and not self.web_comp:
                self.web_comp = 1.0
                reward += 1.20
            else:
                reward += self._invalid()

        elif action == 2:  # recon app
            if self.web_comp and not self.app_disc:
                self.app_disc = 1.0
                reward += 0.55
            else:
                reward += self._invalid()

        elif action == 3:  # exploit app
            if self.app_disc and not self.app_comp:
                self.app_comp = 1.0
                reward += 1.30
                # Defender changes the graph only after the agent has actually
                # progressed onto the primary branch.
                self._decide_intervention()
            else:
                reward += self._invalid()

        elif action == 4:  # app -> db
            if self.app_comp and self.primary_edge and not self.db_comp:
                self.db_comp = 1.0
                self.route = "PRIMARY"
                reward += 2.00
            else:
                reward += self._invalid()

        elif action == 5:  # recon identity
            if self.web_comp and not self.id_disc:
                self.id_disc = 1.0
                # Alternate branch has lower immediate progress reward and
                # therefore higher effective operational cost when primary is open.
                reward += 0.30
            else:
                reward += self._invalid()

        elif action == 6:  # credential access
            if self.id_disc and not self.credential:
                self.credential = 1.0
                reward += 0.65
            else:
                reward += self._invalid()

        elif action == 7:  # identity -> db
            if self.credential and self.alternate_edge and not self.db_comp:
                self.db_comp = 1.0
                self.route = "ALTERNATE"
                reward += 1.35
            else:
                reward += self._invalid()

        elif action == 8:  # objective
            if self.db_comp:
                self.success = True
                reward += 6.0
                terminated = True
            else:
                reward += self._invalid()

        # Mild exposure cost for credential operations. This is not a route-label
        # reward; it represents additional observable activity.
        if action in (6, 7):
            reward -= 0.10

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        info = {
            "success": bool(self.success),
            "route": self.route,
            "steps": int(self.steps),
            "invalid_actions": int(self.invalid),
            "intervention_seen": bool(self.intervention_seen),
            "primary_edge_final": bool(self.primary_edge),
        }
        return self._obs(), float(reward), terminated, truncated, info


def run_episode(model, mode, seed):
    env = DRTTEnv(mode=mode, seed=seed)
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    latencies = []
    actions = []
    done = False

    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        action = int(action)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        done = terminated or truncated

    return {
        "seed": seed,
        "condition": mode,
        "route": info["route"],
        "objective_reached": int(info["success"]),
        "steps": info["steps"],
        "return": total_return,
        "invalid_actions": info["invalid_actions"],
        "intervention_seen": int(info["intervention_seen"]),
        "mean_decision_latency_ms": float(np.mean(latencies)),
        "action_sequence": "-".join(str(x) for x in actions),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100000)
    p.add_argument("--paired-seeds", type=int, default=30)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--output", default="runs/e1_v3_conditional")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.train_seed)
    train_env = Monitor(DRTTEnv("TRAIN", args.train_seed, training_block_prob=0.50))

    print("=" * 76)
    print("E1-v3: PPO CONDITIONAL REROUTING UNDER DYNAMIC GRAPH CHANGE")
    print("=" * 76)
    print("Training timesteps :", args.timesteps)
    print("Training seed      :", args.train_seed)
    print("Paired seeds       :", args.paired_seeds)
    print("Evaluation runs    :", args.paired_seeds * 2)
    print("Output             :", out)
    print("=" * 76)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.train_seed,
        verbose=1,
        device="auto",
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    training_time = time.perf_counter() - t0
    model.save(out / "model")

    rows = []
    for i in range(args.paired_seeds):
        seed = 3000 + i
        rows.append(run_episode(model, "PRIMARY_OPEN", seed))
        rows.append(run_episode(model, "PRIMARY_BLOCKED", seed))

    df = pd.DataFrame(rows)
    df.to_csv(out / "paired_evaluation.csv", index=False)

    op = df[df.condition == "PRIMARY_OPEN"].set_index("seed")
    bl = df[df.condition == "PRIMARY_BLOCKED"].set_index("seed")

    pair_rows = []
    for seed in op.index:
        pair_rows.append({
            "seed": int(seed),
            "open_route": op.loc[seed, "route"],
            "blocked_route": bl.loc[seed, "route"],
            "open_success": int(op.loc[seed, "objective_reached"]),
            "blocked_success": int(bl.loc[seed, "objective_reached"]),
            "primary_to_alternate_switch": int(
                op.loc[seed, "route"] == "PRIMARY"
                and bl.loc[seed, "route"] == "ALTERNATE"
            ),
            "open_steps": int(op.loc[seed, "steps"]),
            "blocked_steps": int(bl.loc[seed, "steps"]),
        })

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "paired_seed_summary.csv", index=False)

    def mean_std(s):
        return [float(s.mean()), float(s.std(ddof=1))]

    summary = {
        "experiment": "E1-v3 PPO Conditional Dynamic-Graph Rerouting",
        "rl_algorithm": "PPO",
        "training_timesteps": args.timesteps,
        "training_seed": args.train_seed,
        "paired_seeds": args.paired_seeds,
        "total_evaluation_runs": 2 * args.paired_seeds,
        "open_success_rate": float(op.objective_reached.mean()),
        "blocked_success_rate": float(bl.objective_reached.mean()),
        "primary_route_rate_when_open": float((op.route == "PRIMARY").mean()),
        "alternate_route_rate_when_primary_blocked": float(
            (bl.route == "ALTERNATE").mean()
        ),
        "paired_primary_to_alternate_switch_rate": float(
            pairs.primary_to_alternate_switch.mean()
        ),
        "open_steps_mean_std": mean_std(op.steps),
        "blocked_steps_mean_std": mean_std(bl.steps),
        "open_invalid_actions_mean_std": mean_std(op.invalid_actions),
        "blocked_invalid_actions_mean_std": mean_std(bl.invalid_actions),
        "open_latency_ms_mean_std": mean_std(op.mean_decision_latency_ms),
        "blocked_latency_ms_mean_std": mean_std(bl.mean_decision_latency_ms),
        "training_time_s": float(training_time),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nE1-v3 FINAL SUMMARY")
    print("=" * 76)
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nPublication-oriented checks")
    print("=" * 76)
    print("Primary route when open      :", summary["primary_route_rate_when_open"])
    print("Alternate when blocked       :", summary["alternate_route_rate_when_primary_blocked"])
    print("Paired conditional switching :", summary["paired_primary_to_alternate_switch_rate"])
    print("Open success                 :", summary["open_success_rate"])
    print("Blocked success              :", summary["blocked_success_rate"])
    print("\nSaved:", out.resolve())


if __name__ == "__main__":
    main()
