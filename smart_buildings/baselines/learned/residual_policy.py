"""Layer B Hierarchical Residual Policy.

Wraps the Layer A physics-aware TransferPolicy with a permutation-invariant
DeepSets ResidualNet that learns bounded setpoint trims without dimensional
incompatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch

from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.learned.residual_net import ResidualNet


class ResidualTransferPolicy(TransferPolicy):
    """Layer B controller: Layer A physics controller + DeepSets residual trims."""

    name = "residual"

    def __init__(
        self,
        building_meta_data: Any,
        action_spec: Any = None,
        weights_path: str | Path | None = None,
        residual_scale: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.residual_scale = residual_scale
        self.device = torch.device("cpu")

        # Initialize network
        self.model = ResidualNet(hidden=64, residual_scale=residual_scale).to(self.device)

        # Attempt to load weights if path provided or if default weights exist
        if weights_path is None:
            default_weights = (
                Path(__file__).resolve().parent / "residual_weights.pt"
            )
            if default_weights.exists():
                weights_path = default_weights

        self.has_weights = False
        if weights_path is not None and Path(weights_path).exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_weights = True
            except Exception as e:
                print(f"[ResidualTransferPolicy] Notice: Could not load weights from {weights_path}: {e}")
                self.has_weights = False
        else:
            self.model.eval()

    def _extract_tensors(
        self,
        obs: np.ndarray,
        nominal_action: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts DeepSets input tensors matching the building's current topology."""
        oat = self._outside_air_k(obs) or 293.15
        oat_norm = (oat - 273.15) / 30.0

        c_now = float(obs[self._comfort_now_idx]) if self._comfort_now_idx is not None else 0.0
        c_soon = float(obs[self._comfort_soon_idx]) if self._comfort_soon_idx is not None else 0.0
        num_occ = float(obs[self._num_occupants_idx]) if self._num_occupants_idx is not None else 0.0
        hod_sin = float(obs[self.hod_sin_index]) if hasattr(self, "hod_sin_index") else 0.0
        hod_cos = float(obs[self.hod_cos_index]) if hasattr(self, "hod_cos_index") else 1.0

        bldg_feat = torch.tensor(
            [c_now, c_soon, hod_sin, hod_cos, oat_norm, num_occ],
            dtype=torch.float32,
            device=self.device,
        )

        zone_rows = []
        zone_to_ahu = []
        low, high = self._active_band(obs)

        for a_idx, ctrl in enumerate(self.ahu_controllers):
            for vav_id, obs_idx, action_indices in ctrl["zones"]:
                lo, hi = self._zone_band(obs, vav_id, low, high)
                temp = self._denormalize_zone_temp(obs[obs_idx])
                err_lo = max(0.0, lo - temp)
                err_hi = max(0.0, temp - hi)

                d_idx = action_indices.get("supply_air_damper_percentage_command")
                r_idx = action_indices.get("reheat_valve_percentage_command")
                d_pos = float(nominal_action["continuous_action"][d_idx]) if d_idx is not None else -1.0
                r_pos = float(nominal_action["continuous_action"][r_idx]) if r_idx is not None else -1.0

                occ_flag = float(c_now > 0.5 or num_occ > 0.0)
                zone_rows.append([err_lo, err_hi, d_pos, r_pos, occ_flag, oat_norm])
                zone_to_ahu.append(a_idx)

        zone_tensor = torch.tensor(zone_rows, dtype=torch.float32, device=self.device)
        zone_to_ahu_tensor = torch.tensor(zone_to_ahu, dtype=torch.int64, device=self.device)

        k = len(self.ahu_controllers)
        ahu_rows = []
        for a_idx, ctrl in enumerate(self.ahu_controllers):
            sat = float(nominal_action["continuous_action"][ctrl["temp_idx"]])
            pidx = ctrl.get("pressure_idx")
            press = float(nominal_action["continuous_action"][pidx]) if pidx is not None else -1.0
            ridx = ctrl.get("recirc_idx")
            rec = float(nominal_action["continuous_action"][ridx]) if ridx is not None else 1.0
            fan_on = float(nominal_action["discrete_action"][ctrl["run_discrete_idx"]])

            n_zones = len(ctrl["zones"])
            child_z = [i for i, a in enumerate(zone_to_ahu) if a == a_idx]
            if child_z:
                child_errs = [zone_rows[i][0] + zone_rows[i][1] for i in child_z]
                mean_e = float(np.mean(child_errs))
                max_e = float(np.max(child_errs))
            else:
                mean_e = 0.0
                max_e = 0.0

            ahu_rows.append([sat, press, rec, fan_on, oat_norm, float(n_zones), mean_e, max_e])

        ahu_tensor = torch.tensor(ahu_rows, dtype=torch.float32, device=self.device)
        return zone_tensor, ahu_tensor, bldg_feat, zone_to_ahu_tensor

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        # Step 1: Compute Layer A nominal physics action
        nominal = super().get_action(obs)

        # If no weights exist, Layer A is used directly
        if not self.has_weights:
            return nominal

        # Step 2: Compute Layer B residual trims
        try:
            with torch.no_grad():
                z_feats, a_feats, b_feats, z2a = self._extract_tensors(obs, nominal)
                residuals = self.model(z_feats, a_feats, b_feats, z2a)

                cont = nominal["continuous_action"].copy()
                disc = nominal["discrete_action"].copy()

                # Apply AHU continuous residuals
                ahu_res = residuals["ahu_cont"].cpu().numpy()
                for i, ctrl in enumerate(self.ahu_controllers):
                    cont[ctrl["temp_idx"]] += ahu_res[i, 0]
                    if ctrl.get("pressure_idx") is not None:
                        cont[ctrl["pressure_idx"]] += ahu_res[i, 1]
                    if ctrl.get("recirc_idx") is not None:
                        cont[ctrl["recirc_idx"]] += ahu_res[i, 2]

                # Apply Hot Water continuous residuals
                water_res = residuals["water_cont"].cpu().numpy()
                if self.water_temp_idx is not None:
                    cont[self.water_temp_idx] += water_res[0]
                if self.water_pressure_idx is not None:
                    cont[self.water_pressure_idx] += water_res[1]

                # Apply Zone continuous residuals
                zone_res = residuals["zone_cont"].cpu().numpy()
                z_idx = 0
                for ctrl in self.ahu_controllers:
                    for _, _, action_indices in ctrl["zones"]:
                        d_idx = action_indices.get("supply_air_damper_percentage_command")
                        r_idx = action_indices.get("reheat_valve_percentage_command")
                        if d_idx is not None:
                            cont[d_idx] += zone_res[z_idx, 0]
                        if r_idx is not None:
                            cont[r_idx] += zone_res[z_idx, 1]
                        z_idx += 1

                # Step 3: Gateway safety bounding
                cont = np.nan_to_num(cont, nan=0.0, posinf=1.0, neginf=-1.0)
                cont = np.clip(cont, self._cont_low, self._cont_high).astype(np.float32)
                disc = np.nan_to_num(disc, nan=0).astype(np.int32)
                disc = np.clip(disc, self._disc_low, self._disc_high).astype(np.int32)

                return {"continuous_action": cont, "discrete_action": disc}
        except Exception as e:
            # Safe fallback to Layer A nominal on any unexpected exception
            return nominal
