import os, sys
from pathlib import Path
os.environ['SMART_BUILDINGS_DATA_ROOT'] = str(Path(".").resolve())
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
sys.path.insert(0, str(Path("scratch/extracted_notebooks").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("neurips-2026-smart-buildings").resolve()))

from scratch.unified_policy import UnifiedMasterPolicy
from scratch.eval_harness import evaluate_policy_on_scenario

scenarios = [
    Path("smart_buildings/training_scenarios/building_train_1/atlanta_2"),
    Path("smart_buildings/training_scenarios/building_train_1/phoenix_5"),
    Path("smart_buildings/training_scenarios/building_train_1/fairbanks_1"),
]

for sc in scenarios:
    print(f"\nEvaluating UnifiedMasterPolicy on {sc.name} (576 steps)...", flush=True)
    evaluate_policy_on_scenario(UnifiedMasterPolicy, sc, steps=576)
