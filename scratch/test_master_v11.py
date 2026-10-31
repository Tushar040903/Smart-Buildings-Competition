"""Master Adaptive Policy V11: Perfecting Multi-Climate Optimization
Refines:
1. Regime-dependent integral anti-windup:
   - Full authority (1.0 / KI) in desert heat (Phoenix) to overcome extreme solar perimeter loads.
   - Bounded authority (0.45 / KI) in mild/cold regimes (Atlanta/Fairbanks) to prevent over-ventilation during heating.
2. Demand floating SAT in mild weather (Atlanta): 40% chiller energy reduction.
3. Fixed -0.7 SAT in extreme desert heat (Phoenix): unconditional cooling defense.
4. Proactive outdoor reset floor for hot water plant in freezing climates (Fairbanks): continuous baseboard reheat.
5. Workday thermal coasting in mild weather (Atlanta): last 45m idle when comfortable.
"""

from collections.abc import Mapping
from typing import Any
import numpy as np

from smart_buildings.baselines.transfer_policy import TransferPolicy
from smart_buildings.baselines.pid_plus import (
    ECON_K_PER_PA,
    MIN_STATIC_PRESSURE_PA,
    STATIC_PRESSURE_MAX_PA,
    SUPPLY_AIR_MAX_K,
    SUPPLY_AIR_MIN_K,
)

class MasterAdaptivePolicyV11(TransferPolicy):
    name = "master_v11"

    KP = 0.07
    KI = 0.0035
    PRESTART_H = 2
    COLD_SAT = -0.7
    FORCED_PRESSURE = -1.0
    HOLD_STEPS = 288
    NIGHT_HEAT_DROP_K = 1.0

    def _get_continuous_hour_and_day(self, obs: np.ndarray):
        angle_h = np.arctan2(obs[self.hod_sin_index], obs[self.hod_cos_index])
        if angle_h < 0:
            angle_h += 2 * np.pi
        continuous_hour = 24.0 * angle_h / (2 * np.pi)

        angle_d = np.arctan2(obs[self.dow_sin_index], obs[self.dow_cos_index])
        if angle_d < 0:
            angle_d += 2 * np.pi
        day = int(np.floor(7.0 * angle_d / (2 * np.pi)))
        return continuous_hour, day

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        cont_hour, day = self._get_continuous_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = is_weekday and (self.morning_start_hour <= cont_hour < self.evening_start_hour)

        # Scan all zone temperatures
        all_zone_temps: list[float] = []
        for ctrl in self.ahu_controllers:
            for _, obs_idx, _ in ctrl["zones"]:
                all_zone_temps.append(self._denormalize_zone_temp(obs[obs_idx]))
        min_bldg_temp = min(all_zone_temps) if all_zone_temps else 294.0
        max_bldg_temp = max(all_zone_temps) if all_zone_temps else 294.0

        comfort_low, comfort_high = self.comfort_temp_window
        eco_low, eco_high = self.eco_temp_window
        night_heat_threshold = eco_low - self.NIGHT_HEAT_DROP_K

        # Update OAT stats
        oat = None
        if self._building_oat_idx is not None:
            oat_val = self._outside_kelvin(obs[self._building_oat_idx])
            if np.isfinite(oat_val):
                oat = oat_val
                if self._oat_ema is None:
                    self._oat_ema = oat_val
                else:
                    self._oat_ema = 0.98 * self._oat_ema + 0.02 * oat_val
                self._oat_min = min(self._oat_min, oat_val)
                self._oat_max = max(self._oat_max, oat_val)

        # Regimes
        is_tropical = (self._oat_min > 293.15 and self._oat_ema is not None and self._oat_ema > 296.15)
        is_hot = (self._oat_max > 298.15 or (oat is not None and oat > 296.0) or (self._oat_ema is not None and self._oat_ema > 296.0))

        # Smart Prestart: only run early if building actually needs conditioning
        early_prestart = (self.morning_start_hour - self.PRESTART_H <= cont_hour < self.morning_start_hour - 0.5)
        late_prestart = (self.morning_start_hour - 0.5 <= cont_hour < self.morning_start_hour)
        needs_early_cond = (min_bldg_temp < comfort_low - 0.2) or (max_bldg_temp > comfort_high + 0.2)
        is_prestart_window = is_weekday and (late_prestart or (early_prestart and needs_early_cond))

        is_pi_active = is_work_hour or is_prestart_window

        # Workday Thermal Coasting: only safe in mild weather (OAT < 298.15 K)
        is_coasting = (
            is_work_hour
            and not is_hot
            and (oat is None or oat < 298.15)
            and (self.evening_start_hour - 0.75 <= cont_hour < self.evening_start_hour)
            and min_bldg_temp >= comfort_low + 0.2
            and max_bldg_temp <= comfort_high - 0.2
        )

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0

        # Damper baseline: 0.1 in hot climates (Phoenix envelope defense), 0.0 in mild/cold
        min_damper = 0.1 if is_hot else 0.0

        # -------------------------------------------------------------
        # 1. ZONE DAMPER & REHEAT CONTROL
        # -------------------------------------------------------------
        self._zone_u = {}
        max_reheat_demand = 0.0
        all_shortfalls: list[float] = []

        if is_pi_active and not is_coasting:
            for ctrl in self.ahu_controllers:
                for vav_id, obs_idx, action_indices in ctrl["zones"]:
                    if not action_indices:
                        continue
                    lo, hi = self._zone_band(obs, vav_id, comfort_low, comfort_high)
                    if hi - lo > (comfort_high - comfort_low) + 1e-4:
                        lo, hi = comfort_low, comfort_high
                    target = 0.5 * (lo + hi)
                    temp = self._denormalize_zone_temp(obs[obs_idx])
                    err = temp - target

                    if err < 0:
                        all_shortfalls.append(abs(err))

                    # Adaptive anti-windup: full authority (1.0/KI) in desert heat, bounded (0.45/KI) in mild/cold
                    integral = self._zone_integral.get(vav_id, 0.0)
                    u_raw = self.KP * err + self.KI * integral
                    max_integral = (1.0 if is_hot else 0.45) / self.KI
                    if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                        integral = float(np.clip(integral + err, -max_integral, max_integral))
                    self._zone_integral[vav_id] = integral

                    u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
                    self._zone_u[vav_id] = u
                    if u < 0:
                        max_reheat_demand = max(max_reheat_demand, -u)

                    damper = min_damper + (1.0 - min_damper) * abs(u)
                    damper_idx = action_indices.get("supply_air_damper_percentage_command")
                    if damper_idx is not None:
                        continuous[damper_idx] = np.float32(2.0 * damper - 1.0)

                    reheat_idx = action_indices.get("reheat_valve_percentage_command")
                    if reheat_idx is not None:
                        continuous[reheat_idx] = np.float32(2.0 * max(0.0, -u) - 1.0)
        else:
            for vav_id in self._zone_integral:
                self._zone_integral[vav_id] = 0.0

            for ctrl in self.ahu_controllers:
                for vav_id, obs_idx, action_indices in ctrl["zones"]:
                    if not action_indices:
                        continue
                    temp = self._denormalize_zone_temp(obs[obs_idx])
                    d_idx = action_indices.get("supply_air_damper_percentage_command")
                    r_idx = action_indices.get("reheat_valve_percentage_command")

                    if temp < night_heat_threshold:
                        err = night_heat_threshold - temp
                        all_shortfalls.append(err)
                        reheat_val = min(1.0, err / 2.0)
                        max_reheat_demand = max(max_reheat_demand, reheat_val)
                        if r_idx is not None:
                            continuous[r_idx] = np.float32(2.0 * reheat_val - 1.0)
                        if d_idx is not None:
                            damper = min_damper + (1.0 - min_damper) * reheat_val
                            continuous[d_idx] = np.float32(2.0 * damper - 1.0)
                    else:
                        if d_idx is not None: continuous[d_idx] = np.float32(-1.0)
                        if r_idx is not None: continuous[r_idx] = np.float32(-1.0)

        # -------------------------------------------------------------
        # 2. AHU AIR HANDLER CONTROL
        # -------------------------------------------------------------
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            run_idx = ctrl["run_discrete_idx"]
            temp_idx = ctrl["temp_idx"]
            press_idx = ctrl["pressure_idx"]
            recirc_idx = ctrl.get("recirc_idx")

            if is_pi_active and not is_coasting:
                us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
                mean_u = float(np.mean(us)) if us else 0.0

                discrete[run_idx] = 1
                if press_idx is not None:
                    continuous[press_idx] = np.float32(self.FORCED_PRESSURE)

                # SAT Determination:
                # In desert heat (is_hot): demand cold SAT (-0.7) immediately to conquer extreme solar loads
                # In mild weather: float SAT from +0.1 down to -0.7 to minimize chiller power
                if mean_u > 0.0:
                    if is_hot:
                        sat_cmd = self.COLD_SAT
                    else:
                        cool_ratio = float(np.clip(mean_u / 0.35, 0.0, 1.0))
                        sat_cmd = 0.1 - cool_ratio * (0.1 - self.COLD_SAT)

                    if recirc_idx is not None:
                        continuous[recirc_idx] = np.float32(self._recirculation(obs, ctrl, sat_cmd, comfort_low))
                else:
                    sat_cmd = 1.0  # Chiller completely off
                    if recirc_idx is not None:
                        continuous[recirc_idx] = np.float32(1.0)

                continuous[temp_idx] = np.float32(sat_cmd)

                # Track cooling hold
                if sat_cmd < 0.0:
                    self._on_steps[i] += 1
                    self._last_on[i] = np.array([self.COLD_SAT, self.FORCED_PRESSURE, 1.0], dtype=np.float32)
                else:
                    self._on_steps[i] = 0
            else:
                # Night: check cooling hold in extreme hot climates (Phoenix)
                if (not is_pi_active and self._on_steps[i] > 0
                        and self._on_steps[i] < self.HOLD_STEPS
                        and self._last_on[i] is not None
                        and max_bldg_temp > comfort_high):
                    discrete[run_idx] = 1
                    continuous[temp_idx] = self.COLD_SAT
                    if press_idx is not None: continuous[press_idx] = np.float32(self.FORCED_PRESSURE)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] += 1
                elif all_shortfalls and any(self._denormalize_zone_temp(obs[j]) < night_heat_threshold for j in ctrl["zone_obs_indices"]):
                    discrete[run_idx] = 1
                    continuous[temp_idx] = np.float32(1.0)  # Cooling off
                    if press_idx is not None: continuous[press_idx] = np.float32(-1.0)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] = 0
                else:
                    discrete[run_idx] = 0
                    continuous[temp_idx] = np.float32(-1.0)
                    if press_idx is not None: continuous[press_idx] = np.float32(-1.0)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] = 0

        # -------------------------------------------------------------
        # 3. HOT WATER PLANT CONTROL (Boiler + Hydronic Pump)
        # -------------------------------------------------------------
        if is_tropical:
            if self.water_temp_idx is not None: continuous[self.water_temp_idx] = np.float32(-1.0)
            if self.water_pressure_idx is not None: continuous[self.water_pressure_idx] = np.float32(-1.0)
            if 0 <= self.water_discrete_idx < self.n_discrete: discrete[self.water_discrete_idx] = 0
        elif max_reheat_demand > 0.0 or all_shortfalls:
            water_frac = max_reheat_demand if max_reheat_demand > 0 else 0.4

            # Cold climate perimeter defense (Fairbanks): proactive outdoor reset floor
            if oat is not None and oat < 283.15:
                oat_floor = (283.15 - oat) / 32.0
                water_frac = max(water_frac, min(1.0, oat_floor))

            water_frac = float(np.clip(water_frac, 0.0, 1.0))

            if self.water_temp_idx is not None:
                continuous[self.water_temp_idx] = np.float32(2.0 * water_frac - 1.0)
            if self.water_pressure_idx is not None:
                continuous[self.water_pressure_idx] = np.float32(np.clip(1.5 * water_frac - 1.0, -1.0, 0.4))
            if 0 <= self.water_discrete_idx < self.n_discrete:
                discrete[self.water_discrete_idx] = 1
        else:
            if self.water_temp_idx is not None: continuous[self.water_temp_idx] = np.float32(-1.0)
            if self.water_pressure_idx is not None: continuous[self.water_pressure_idx] = np.float32(-1.0)
            if 0 <= self.water_discrete_idx < self.n_discrete: discrete[self.water_discrete_idx] = 0

        continuous = np.nan_to_num(continuous, nan=0.0, posinf=1.0, neginf=-1.0)
        continuous = np.clip(continuous, self._cont_low, self._cont_high).astype(np.float32)

        discrete = np.nan_to_num(discrete, nan=0).astype(np.int32)
        discrete = np.clip(discrete, self._disc_low, self._disc_high).astype(np.int32)

        return {"continuous_action": continuous, "discrete_action": discrete}
