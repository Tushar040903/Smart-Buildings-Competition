import os, sys
from pathlib import Path
os.environ['SMART_BUILDINGS_DATA_ROOT'] = str(Path(".").resolve())
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
sys.path.insert(0, str(Path("scratch/extracted_notebooks").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("neurips-2026-smart-buildings").resolve()))

from scratch.unified_policy import UnifiedMasterPolicy
from smart_buildings.baselines.transfer_policy import TransferPolicy
from importlib import import_module
k07_mod = import_module("smart-buildings-probe-ds-k07")
CompetitorPolicy = k07_mod.SubmissionPolicy

from scratch.eval_harness import evaluate_policy_on_scenario

scenarios = [
    Path("smart_buildings/training_scenarios/building_train_1/atlanta_2"),
    Path("smart_buildings/training_scenarios/building_train_1/phoenix_5"),
    Path("smart_buildings/training_scenarios/building_train_1/fairbanks_1"),
]

for sc in scenarios:
    print(f"\n{'='*55}\nSCENARIO: {sc.name} (576 steps)")
    print("--- 1. Our Previous TransferPolicy ---")
    evaluate_policy_on_scenario(TransferPolicy, sc, steps=576)
    print("--- 2. Competitor SubmissionPolicy (0.01708) ---")
    evaluate_policy_on_scenario(CompetitorPolicy, sc, steps=576)
    print("--- 3. NEW UnifiedMasterPolicy (Synthesized) ---")
    evaluate_policy_on_scenario(UnifiedMasterPolicy, sc, steps=576)
