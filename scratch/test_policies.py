import os, sys
from pathlib import Path
os.environ['SMART_BUILDINGS_DATA_ROOT'] = str(Path(".").resolve())
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
sys.path.insert(0, str(Path("scratch/extracted_notebooks").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("neurips-2026-smart-buildings").resolve()))

# Import competitor policy
from importlib import import_module
k07_mod = import_module("smart-buildings-probe-ds-k07")
CompetitorPolicy = k07_mod.SubmissionPolicy

# Import our policy
from smart_buildings.baselines.transfer_policy import TransferPolicy
from scratch.eval_harness import evaluate_policy_on_scenario

scenario = Path("smart_buildings/training_scenarios/building_train_1/atlanta_2")

print("Testing on 576 steps (2 full days)...")
print("1. Evaluating our current TransferPolicy:")
evaluate_policy_on_scenario(TransferPolicy, scenario, steps=576)

print("2. Evaluating Competitor Policy (Score 0.01708):")
evaluate_policy_on_scenario(CompetitorPolicy, scenario, steps=576)
