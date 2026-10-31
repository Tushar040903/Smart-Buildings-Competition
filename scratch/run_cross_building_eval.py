"""Run 576-step evaluation on building_train_1 and building_train_2 across all 5 climates.
Compares PIDWaterResetPolicy (official baseline) vs TransferPolicy (ours, Run 5).
Also extracts exact reward component breakdowns using get_reward_info_and_response().
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

STEPS = 576

scenarios = [
    ("Building 1", "Atlanta", "building_train_1/atlanta_2"),
    ("Building 1", "Denver", "building_train_1/denver_7"),
    ("Building 1", "Fairbanks", "building_train_1/fairbanks_1"),
    ("Building 1", "Phoenix", "building_train_1/phoenix_5"),
    ("Building 1", "Singapore", "building_train_1/singapore_8"),
    ("Building 2", "Atlanta", "building_train_2/atlanta_3"),
    ("Building 2", "Denver", "building_train_2/denver_8"),
    ("Building 2", "Fairbanks", "building_train_2/fairbanks_2"),
    ("Building 2", "Phoenix", "building_train_2/phoenix_6"),
    ("Building 2", "Singapore", "building_train_2/singapore_1"),
]

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

def eval_policy(cfg_path, sc_path, policy_cls):
    env = load_hybrid_environment(cfg_path)
    meta = create_building_meta_data(env, sc_path)
    action_spec = create_action_spec(meta)
    policy = policy_cls(meta, action_spec)
    obs, _ = env.reset()

    total_r = 0.0
    tot_prod = 0.0
    tot_gas = 0.0
    tot_elec = 0.0
    tot_carb = 0.0

    for _ in range(STEPS):
        act = policy.get_action(obs)
        obs, reward, term, trunc, info = env.step(act)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        total_r += r

        _, resp = env.get_reward_info_and_response()
        tot_prod += resp.productivity_reward
        tot_gas += resp.natural_gas_energy_cost
        tot_elec += resp.electricity_energy_cost
        tot_carb += resp.carbon_cost

    return {
        "total_reward": total_r,
        "mean_reward": total_r / STEPS,
        "productivity": tot_prod,
        "gas_cost": tot_gas,
        "elec_cost": tot_elec,
        "carbon_cost": tot_carb,
    }

def main():
    print(f"{'Building':<12} | {'Climate':<10} | {'Baseline (pid_water)':<20} | {'Ours (TransferPolicy)':<20} | {'Improvement':<12}", flush=True)
    print("-" * 84, flush=True)

    results = []

    for bldg, climate, sc_rel in scenarios:
        sc_path = Path("smart_buildings/training_scenarios") / sc_rel
        cfg_path = sc_path / "config.gin"

        res_base = eval_policy(cfg_path, sc_path, PIDWaterResetPolicy)
        res_ours = eval_policy(cfg_path, sc_path, TransferPolicy)

        r_b = res_base["total_reward"]
        r_o = res_ours["total_reward"]
        imp = ((r_b - r_o) / abs(r_b)) * 100.0 if abs(r_b) > 1e-6 else 0.0

        print(f"{bldg:<12} | {climate:<10} | {r_b:9.2f} (mean {r_b/STEPS:7.4f}) | {r_o:9.2f} (mean {r_o/STEPS:7.4f}) | {imp:+9.1f}%", flush=True)

        results.append({
            "building": bldg,
            "climate": climate,
            "baseline": res_base,
            "ours": res_ours,
            "improvement": imp,
        })

    # Summary
    print("\n" + "=" * 84, flush=True)
    b1_base = np.mean([r["baseline"]["total_reward"] for r in results if r["building"] == "Building 1"])
    b1_ours = np.mean([r["ours"]["total_reward"] for r in results if r["building"] == "Building 1"])
    b1_imp = ((b1_base - b1_ours) / abs(b1_base)) * 100.0

    b2_base = np.mean([r["baseline"]["total_reward"] for r in results if r["building"] == "Building 2"])
    b2_ours = np.mean([r["ours"]["total_reward"] for r in results if r["building"] == "Building 2"])
    b2_imp = ((b2_base - b2_ours) / abs(b2_base)) * 100.0

    all_base = np.mean([r["baseline"]["total_reward"] for r in results])
    all_ours = np.mean([r["ours"]["total_reward"] for r in results])
    all_imp = ((all_base - all_ours) / abs(all_base)) * 100.0

    print(f"Building 1 Average: Baseline={b1_base:.2f} | Ours={b1_ours:.2f} | Improvement={b1_imp:+.1f}%", flush=True)
    print(f"Building 2 Average: Baseline={b2_base:.2f} | Ours={b2_ours:.2f} | Improvement={b2_imp:+.1f}%", flush=True)
    print(f"Overall Average:    Baseline={all_base:.2f} | Ours={all_ours:.2f} | Improvement={all_imp:+.1f}%", flush=True)

if __name__ == "__main__":
    main()

