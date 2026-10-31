import os, sys
from pathlib import Path
os.environ['SMART_BUILDINGS_DATA_ROOT'] = str(Path(".").resolve())
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
sys.path.insert(0, str(Path("scratch/extracted_notebooks").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("neurips-2026-smart-buildings").resolve()))

from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy
from scratch.unified_policy import UnifiedMasterPolicy
import numpy as np

scenario_path = Path("smart_buildings/training_scenarios/building_train_1/fairbanks_1")
env = load_hybrid_environment(scenario_path / "config.gin")
env.metrics_path = None
meta = create_building_meta_data(env, scenario_path)

cont_dim = meta["n_continuous_action"]
disc_dim = meta["n_discrete_action"]
class DummySpace:
    def __init__(self, low, high, dtype):
        self.low = low
        self.high = high
        self.dtype = dtype
action_spec = {
    "continuous_action": DummySpace(np.full(cont_dim, -1.0, dtype=np.float32), np.full(cont_dim, 1.0, dtype=np.float32), np.float32),
    "discrete_action": DummySpace(np.zeros(disc_dim, dtype=np.int32), np.ones(disc_dim, dtype=np.int32), np.int32),
}

def debug_policy(policy_cls, name):
    p = policy_cls(meta, action_spec)
    obs, _ = env.reset()
    print(f"\n--- DEBUG {name} on fairbanks_1 (first 20 steps) ---")
    tot_r = 0.0
    for step in range(20):
        action = p.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        tot_r += r
        hw_idx = getattr(p, 'water_temp_idx', None)
        hw_t = action["continuous_action"][hw_idx] if hw_idx is not None else -1.0
        hw_disc_idx = getattr(p, 'water_discrete_idx', None)
        hw_disc = action["discrete_action"][hw_disc_idx] if hw_disc_idx is not None else -1
        # zone temps
        z_temps = [p._denormalize_zone_temp(obs[j]) for ctrl in p.ahu_controllers for _, j, _ in ctrl["zones"]]
        print(f"step {step:02d}: r={r:+.5f}, min_T={min(z_temps):.1f}K, mean_T={np.mean(z_temps):.1f}K, max_T={max(z_temps):.1f}K, hw_cmd={hw_t:.2f}, hw_disc={hw_disc}")
    print(f"Total 20-step reward: {tot_r:.4f}")

debug_policy(TransferPolicy, "TransferPolicy")
debug_policy(UnifiedMasterPolicy, "UnifiedMasterPolicy")
