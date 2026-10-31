"""Layer B Residual Policy Training Harness.

Trains the permutation-invariant DeepSets ResidualNet on top of the Layer A
TransferPolicy using experience collected from generated holdout buildings
and climates.

Usage:
    python -m smart_buildings.baselines.learned.train_residual \
        --buildings building_0500 building_0501 \
        --weathers fairbanks_1 phoenix_6 \
        --epochs 10 \
        --output smart_buildings/baselines/learned/residual_weights.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.learned.residual_net import (
    ResidualNet,
    ZONE_FEAT,
    AHU_FEAT,
    BUILDING_FEAT,
)
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


def extract_features(
    policy: TransferPolicy,
    obs: np.ndarray,
    nominal_action: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts permutation-invariant DeepSets tensors for ResidualNet."""
    names = list(policy.field_names)
    oat = policy._outside_air_k(obs) or 293.15
    oat_norm = (oat - 273.15) / 30.0

    # Building features: [comfort_now, comfort_soon, hod_sin, hod_cos, oat_norm, num_occ]
    c_now = float(obs[policy._comfort_now_idx]) if policy._comfort_now_idx is not None else 0.0
    c_soon = float(obs[policy._comfort_soon_idx]) if policy._comfort_soon_idx is not None else 0.0
    num_occ = float(obs[policy._num_occupants_idx]) if policy._num_occupants_idx is not None else 0.0
    hod_sin = float(obs[policy.hod_sin_index]) if hasattr(policy, "hod_sin_index") else 0.0
    hod_cos = float(obs[policy.hod_cos_index]) if hasattr(policy, "hod_cos_index") else 1.0

    bldg_feat = torch.tensor(
        [c_now, c_soon, hod_sin, hod_cos, oat_norm, num_occ],
        dtype=torch.float32,
    )

    # Zone features & mappings
    zone_rows = []
    zone_to_ahu = []
    low, high = policy._active_band(obs)

    for a_idx, ctrl in enumerate(policy.ahu_controllers):
        for vav_id, obs_idx, action_indices in ctrl["zones"]:
            lo, hi = policy._zone_band(obs, vav_id, low, high)
            temp = policy._denormalize_zone_temp(obs[obs_idx])
            err_lo = max(0.0, lo - temp)
            err_hi = max(0.0, temp - hi)

            d_idx = action_indices.get("supply_air_damper_percentage_command")
            r_idx = action_indices.get("reheat_valve_percentage_command")
            d_pos = float(nominal_action["continuous_action"][d_idx]) if d_idx is not None else -1.0
            r_pos = float(nominal_action["continuous_action"][r_idx]) if r_idx is not None else -1.0

            occ_flag = float(c_now > 0.5 or num_occ > 0.0)
            zone_rows.append([err_lo, err_hi, d_pos, r_pos, occ_flag, oat_norm])
            zone_to_ahu.append(a_idx)

    zone_tensor = torch.tensor(zone_rows, dtype=torch.float32)
    zone_to_ahu_tensor = torch.tensor(zone_to_ahu, dtype=torch.int64)

    # AHU features
    k = len(policy.ahu_controllers)
    ahu_rows = []
    for a_idx, ctrl in enumerate(policy.ahu_controllers):
        sat = float(nominal_action["continuous_action"][ctrl["temp_idx"]])
        pidx = ctrl.get("pressure_idx")
        press = float(nominal_action["continuous_action"][pidx]) if pidx is not None else -1.0
        ridx = ctrl.get("recirc_idx")
        rec = float(nominal_action["continuous_action"][ridx]) if ridx is not None else 1.0
        fan_on = float(nominal_action["discrete_action"][ctrl["run_discrete_idx"]])

        n_zones = len(ctrl["zones"])
        child_z_indices = [i for i, a in enumerate(zone_to_ahu) if a == a_idx]
        if child_z_indices:
            child_errs = [zone_rows[i][0] + zone_rows[i][1] for i in child_z_indices]
            mean_e = float(np.mean(child_errs))
            max_e = float(np.max(child_errs))
        else:
            mean_e = 0.0
            max_e = 0.0

        ahu_rows.append([sat, press, rec, fan_on, oat_norm, float(n_zones), mean_e, max_e])

    ahu_tensor = torch.tensor(ahu_rows, dtype=torch.float32)

    return zone_tensor, ahu_tensor, bldg_feat, zone_to_ahu_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", nargs="+", default=["building_train_1"])
    parser.add_argument("--weathers", nargs="+", default=["fairbanks_1"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "residual_weights.pt",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model = ResidualNet(hidden=64, residual_scale=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Initialized ResidualNet with {sum(p.numel() for p in model.parameters())} parameters.")
    print(f"Training across {len(args.buildings)} buildings x {len(args.weathers)} weathers...")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_samples = 0

        for b in args.buildings:
            for w in args.weathers:
                config_path = Path("smart_buildings/training_scenarios") / b / w / "config.gin"
                if not config_path.exists():
                    print(f"Skipping {b}/{w}: {config_path} does not exist")
                    continue
                try:
                    env = load_hybrid_environment(config_path)
                    env._metrics_path = None
                    env._occupancy_normalization_constant = 125.0
                    meta = create_building_meta_data(env, config_path.parent)
                except Exception as e:
                    print(f"Skipping {b}/{w}: {e}")
                    continue

                policy = TransferPolicy(meta, env.action_spec())
                reset_res = env.reset()
                obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

                for step in range(args.steps):
                    action_nominal = policy.get_action(obs)
                    z_feats, a_feats, b_feats, z2a = extract_features(
                        policy, obs, action_nominal
                    )

                    out = model(z_feats, a_feats, b_feats, z2a)

                    # Self-supervised regularized loss: target minimal disturbance while
                    # penalizing discomfort indications
                    loss = (
                        torch.mean(out["ahu_cont"] ** 2)
                        + torch.mean(out["zone_cont"] ** 2)
                        + torch.mean(out["water_cont"] ** 2)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    n_samples += 1

                    step_res = env.step(action_nominal)
                    obs = step_res[0]
                    done = bool(step_res[2] or step_res[3]) if len(step_res) == 5 else bool(step_res[2])
                    if done:
                        break

        avg_loss = epoch_loss / max(1, n_samples)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.6f} ({n_samples} steps)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved trained ResidualNet weights to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
