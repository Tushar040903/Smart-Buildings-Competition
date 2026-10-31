"""Benchmark all training buildings across all climate zones.
Compares PIDWaterResetPolicy (official baseline) vs TransferPolicy (ours).
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

BUILDINGS = [
    "building_train_1",
    "building_train_2",
    "building_train_4",
    "building_train_5",
    "building_train_7",
    "building_train_8",
]

CLIMATES = ["atlanta", "phoenix", "fairbanks", "denver", "singapore"]

STEPS = 576

class DummySpace:
    def __init__(self, low, high, dtype):
        self.low, self.high, self.dtype = low, high, dtype

def create_action_spec(meta):
    return {
        "continuous_action": DummySpace(
            np.full(meta["n_continuous_action"], -1.0, dtype=np.float32),
            np.full(meta["n_continuous_action"], 1.0, dtype=np.float32),
            np.float32,
        ),
        "discrete_action": DummySpace(
            np.zeros(meta["n_discrete_action"], dtype=np.int32),
            np.ones(meta["n_discrete_action"], dtype=np.int32),
            np.int32,
        ),
    }

def evaluate_policy(env, meta, policy_cls, steps=STEPS):
    action_spec = create_action_spec(meta)
    policy = policy_cls(meta, action_spec)
    obs, _ = env.reset()
    total_r = 0.0
    for _ in range(steps):
        act = policy.get_action(obs)
        obs, reward, term, trunc, info = env.step(act)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        total_r += r
    return total_r

def main():
    base_dir = Path("smart_buildings/training_scenarios")
    results = {}

    print(f"Starting Multi-Building Benchmark (steps={STEPS})...", flush=True)
    t0 = time.time()

    for b in BUILDINGS:
        b_dir = base_dir / b
        if not b_dir.exists():
            continue
        scenarios = sorted([d.name for d in b_dir.iterdir() if d.is_dir()])
        print(f"\n=================== {b} ===================", flush=True)
        results[b] = {}

        for sc in scenarios:
            sc_path = b_dir / sc
            cfg_path = sc_path / "config.gin"
            if not cfg_path.exists():
                continue

            # Identify climate tag
            tag = "unknown"
            for c in CLIMATES:
                if sc.startswith(c):
                    tag = c
                    break

            try:
                env = load_hybrid_environment(cfg_path)
                meta = create_building_meta_data(env, sc_path)

                r_base = evaluate_policy(env, meta, PIDWaterResetPolicy, steps=STEPS)
                r_ours = evaluate_policy(env, meta, TransferPolicy, steps=STEPS)

                diff = r_ours - r_base
                pct = ((r_base - r_ours) / abs(r_base) * 100.0) if abs(r_base) > 1e-6 else 0.0

                results[b][tag] = {
                    "scenario": sc,
                    "baseline": r_base,
                    "ours": r_ours,
                    "diff": diff,
                    "improvement_pct": pct,
                }
                print(f"  [{sc:<12}] Baseline: {r_base:7.2f} | Ours: {r_ours:7.2f} | Diff: {diff:+6.2f} ({pct:+5.1f}%)", flush=True)
            except Exception as e:
                print(f"  [{sc:<12}] ERROR: {e}", flush=True)

    print("\n\n=================== SUMMARY TABLE ===================", flush=True)
    print(f"{'Building':<18} | {'Baseline Mean':<14} | {'Ours Mean':<14} | {'Advantage (%)':<14}", flush=True)
    print("-" * 68, flush=True)

    all_base, all_ours = [], []
    for b, sc_dict in results.items():
        if not sc_dict:
            continue
        b_base = np.mean([v["baseline"] for v in sc_dict.values()])
        b_ours = np.mean([v["ours"] for v in sc_dict.values()])
        adv = ((b_base - b_ours) / abs(b_base)) * 100.0 if abs(b_base) > 1e-6 else 0.0
        all_base.append(b_base)
        all_ours.append(b_ours)
        print(f"{b:<18} | {b_base:14.2f} | {b_ours:14.2f} | {adv:+13.1f}%", flush=True)

    total_base_mean = np.mean(all_base)
    total_ours_mean = np.mean(all_ours)
    total_adv = ((total_base_mean - total_ours_mean) / abs(total_base_mean)) * 100.0
    print("-" * 68, flush=True)
    print(f"{'OVERALL AVERAGE':<18} | {total_base_mean:14.2f} | {total_ours_mean:14.2f} | {total_adv:+13.1f}%", flush=True)
    print(f"\nTotal benchmark time: {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()

