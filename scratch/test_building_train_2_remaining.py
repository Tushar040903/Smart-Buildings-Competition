from pathlib import Path
import numpy as np
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

STEPS = 576

scenarios = [
    ("Denver", "building_train_2/denver_8"),
    ("Singapore", "building_train_2/singapore_1"),
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
    for _ in range(STEPS):
        act = policy.get_action(obs)
        obs, reward, term, trunc, info = env.step(act)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        total_r += r
    return total_r

for name, sc_rel in scenarios:
    sc_path = Path("smart_buildings/training_scenarios") / sc_rel
    cfg = sc_path / "config.gin"
    r_base = eval_policy(cfg, sc_path, PIDWaterResetPolicy)
    r_ours = eval_policy(cfg, sc_path, TransferPolicy)
    print(f"[{name}] Baseline: {r_base:.3f} | Ours: {r_ours:.3f} | Diff: {r_ours - r_base:+.3f}")

