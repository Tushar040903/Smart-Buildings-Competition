import time
import numpy as np
from pathlib import Path
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data

def evaluate_policy_on_scenario(policy_cls, scenario_path, steps=8641, **policy_kwargs):
    scenario_path = Path(scenario_path)
    env = load_hybrid_environment(scenario_path / "config.gin")
    env.metrics_path = None
    meta = create_building_meta_data(env, scenario_path)
    
    # Fake action_spec
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
    
    policy = policy_cls(meta, action_spec, **policy_kwargs)
    obs, _ = env.reset()
    
    total_reward = 0.0
    t0 = time.time()
    
    for step in range(steps):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        r = reward.get("agent_0", 0.0) if isinstance(reward, dict) else float(reward)
        total_reward += r
        if terminated or truncated:
            break
            
    elapsed = time.time() - t0
    mean_reward = total_reward / (step + 1)
    print(f"[{policy_cls.__name__}] {scenario_path.name}: total={total_reward:.2f}, mean={mean_reward:.5f}, steps={step+1}, time={elapsed:.1f}s")
    return total_reward, mean_reward
