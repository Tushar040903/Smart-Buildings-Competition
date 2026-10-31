import numpy as np
from pathlib import Path
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy

scenarios = [
    ("Fairbanks", "building_train_1/fairbanks_1"),
    ("Phoenix", "building_train_1/phoenix_5"),
    ("Atlanta", "building_train_1/atlanta_2"),
]

for label, sc_rel in scenarios:
    sc_path = Path("smart_buildings/training_scenarios") / sc_rel
    env = load_hybrid_environment(sc_path / "config.gin")
    meta = create_building_meta_data(env, sc_path)

    class DummySpace:
        def __init__(self, low, high, dtype):
            self.low, self.high, self.dtype = low, high, dtype

    action_spec = {
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

    policy = TransferPolicy(meta, action_spec)
    obs, _ = env.reset()

    total_r = 0.0
    total_comfort_pen = 0.0
    total_energy_cost = 0.0
    total_carbon = 0.0
    violation_steps = 0
    max_comfort_pen = 0.0

    for step in range(576):
        act = policy.get_action(obs)
        obs, reward, term, trunc, info = env.step(act)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        total_r += r

        # inspect info dict
        # typical info keys: 'comfort_regret', 'energy_cost', 'carbon_emission', etc.
        # let's look at info structure
        if step == 0 and label == "Fairbanks":
            print(f"Sample info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
            if isinstance(info, dict):
                for k, v in info.items():
                    print(f"  {k}: {v}")

        if isinstance(info, dict):
            # check agent_0 or direct keys
            agent_info = info.get("agent_0", info)
            p = agent_info.get("occupant_thermal_regret", agent_info.get("comfort_regret", 0.0))
            c = agent_info.get("energy_cost", 0.0)
            e = agent_info.get("carbon_emission", 0.0)
            total_comfort_pen += p
            total_energy_cost += c
            total_carbon += e
            if p > 1e-4:
                violation_steps += 1
                max_comfort_pen = max(max_comfort_pen, p)

    print(f"\n=== {label} (576 steps) ===")
    print(f"Total reward: {total_r:.3f} (mean: {total_r/576:.5f})")
    print(f"Comfort regret sum: {total_comfort_pen:.3f} (violations in {violation_steps}/576 steps, max={max_comfort_pen:.4f})")
    print(f"Energy cost sum: {total_energy_cost:.3f}")
    print(f"Carbon emission sum: {total_carbon:.3f}")
