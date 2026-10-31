import numpy as np
from pathlib import Path
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from smart_buildings.baselines.transfer_policy import TransferPolicy
from scratch.unified_v9 import UnifiedMasterPolicyV9

sc_path = Path('smart_buildings/training_scenarios/building_train_1/phoenix_5')

env1 = load_hybrid_environment(sc_path / 'config.gin')
meta1 = create_building_meta_data(env1, sc_path)

class DummySpace:
    def __init__(self, low, high, dtype):
        self.low, self.high, self.dtype = low, high, dtype

action_spec = {
    'continuous_action': DummySpace(np.full(meta1['n_continuous_action'], -1.0, dtype=np.float32), np.full(meta1['n_continuous_action'], 1.0, dtype=np.float32), np.float32),
    'discrete_action': DummySpace(np.zeros(meta1['n_discrete_action'], dtype=np.int32), np.ones(meta1['n_discrete_action'], dtype=np.int32), np.int32),
}

p1 = TransferPolicy(meta1, action_spec)
p2 = UnifiedMasterPolicyV9(meta1, action_spec)

obs1, _ = env1.reset()
env2 = load_hybrid_environment(sc_path / 'config.gin')
obs2, _ = env2.reset()

for s in range(320):
    a1 = p1.get_action(obs1)
    a2 = p2.get_action(obs2)
    if s in [304, 305, 306, 307]:
        print(f"=== Step {s} (Hour {(s % 288)*5/60:.2f}) ===")
        print(f"p1 on_steps: {p1._on_steps}, p2 on_steps: {p2._on_steps}")
        print(f"p1 discrete: {a1['discrete_action'][:4]}")
        print(f"p2 discrete: {a2['discrete_action'][:4]}")
        t_indices = [c['temp_idx'] for c in p1.ahu_controllers]
        print(f"p1 SAT: {a1['continuous_action'][t_indices]}")
        print(f"p2 SAT: {a2['continuous_action'][t_indices]}")
        # check zone dampers
        d_indices = [ctrl['zones'][0][2]['supply_air_damper_percentage_command'] for ctrl in p1.ahu_controllers if 'supply_air_damper_percentage_command' in ctrl['zones'][0][2]]
        print(f"p1 Dampers[0]: {a1['continuous_action'][d_indices[:3]]}")
        print(f"p2 Dampers[0]: {a2['continuous_action'][d_indices[:3]]}")
    obs1, r1, _, _, _ = env1.step(a1)
    obs2, r2, _, _, _ = env2.step(a2)
    if s in [304, 305, 306, 307]:
        print(f"Reward p1: {r1:.5f}, p2: {r2:.5f}\n")
