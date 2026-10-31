"""Unified Master HVAC Controller (Physics-Aware Hierarchical + Per-Zone PI + Demand SAT).

Synthesizes:
1. Occupied / Prestart: Per-Zone PI to band centre (Kp=0.07, Ki=0.0035, conditional anti-windup)
2. AHU Demand SAT: cooling off (+1.0) when zones need heat, cold (COLD_SAT=-0.7) when zones need cooling
3. Cooling-run hold (minimum on-time to prevent AHU cycling in hot climates)
4. Working economizer with raw-kelvin outdoor air temperature handling
5. Deep Standby unoccupied energy suppression (shutting off AHUs/pumps when inside eco band)
6. Night heating maintenance: if zones drop below eco heating setpoint (active_low), reheat + hot water protects building mass
7. Tropical climate heating lockout (100% gas/pump savings in Singapore)
8. Cold-climate perimeter defense & modulated hydronic pump differential pressure (Fairbanks)
9. Workday thermal coasting
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np

# Native physical simulator bounds
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
MIN_STATIC_PRESSURE_PA = 200.0
ECON_K_PER_PA = 0.00322

OVERHEAT_GUARD_K = 303.15  # 30 C


class UnifiedMasterPolicy:
    name = "unified_master"

    # Per-zone PI parameters
    KP = 0.07
    KI = 0.0035
    PRESTART_H = 2
    MIN_DAMPER = 0.1
    COLD_SAT = -0.7        # ~15 C supply air when cooling demanded
    FORCED_PRESSURE = -1.0  # static pressure floor (200 Pa) during forced/mild runs
    HOLD_STEPS = 288       # 24-hour cooling hold in warm climates
    NIGHT_COOL_CEILING = True
    NIGHT_HEAT_DROP_K = 1.0  # Allow 1 K setback below eco heating before engaging night heating

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None):
        self.metadata = building_meta_data
        self.action_spec = action_spec

        self.comfort_temp_window = tuple(building_meta_data["comfort_temp_window"])
        self.eco_temp_window = tuple(building_meta_data["eco_temp_window"])
        self.morning_start_hour = float(building_meta_data["morning_start_hour"])
        self.evening_start_hour = float(building_meta_data["evening_start_hour"])

        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.n_continuous = int(building_meta_data["n_continuous_action"])
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.zone_action_indices = building_meta_data["zone_action_indices"]
        self.field_names = list(building_meta_data["observation_field_names"])
        self.ahu_zone_assignments = building_meta_data["ahu_zone_assignments"]

        norm_meta = building_meta_data.get("observation_normalization_metadata", {})
        zone_norm = norm_meta.get("zone_air_temperature_sensor", {})
        self.zone_temp_mean = float(zone_norm.get("sample_mean", 294.0))
        self.zone_temp_std = float(zone_norm.get("sample_std", 5.0))

        oat_norm = norm_meta.get("outside_air_temperature_sensor", {})
        self.oat_mean = float(oat_norm.get("sample_mean", 285.0))
        self.oat_std = float(oat_norm.get("sample_std", 10.0))

        self.hod_sin_index = self.field_names.index("hod_sin_000")
        self.hod_cos_index = self.field_names.index("hod_cos_000")
        self.dow_sin_index = self.field_names.index("dow_sin_000")
        self.dow_cos_index = self.field_names.index("dow_cos_000")

        names = self.field_names
        self._comfort_now_idx = names.index("comfort_mode_now") if "comfort_mode_now" in names else None
        self._comfort_soon_idx = names.index("comfort_mode_soon") if "comfort_mode_soon" in names else None
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None

        self._build_mappings()
        self.reset()

    def _build_mappings(self) -> None:
        names = self.field_names
        zone_temp_field_index = {
            name: i for i, name in enumerate(names) if "zone_air_temperature" in name
        }

        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=lambda s: int("".join(filter(str.isdigit, s)) or 0))
        self.n_ahu = min(len(self.temp_indices), len(ahu_ids))
        ahu_ids = ahu_ids[: self.n_ahu]

        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            zones: list[tuple[str, int, dict[str, int]]] = []
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                vav_field = vav_id + "_zone_air_temperature_sensor"
                if vav_field in zone_temp_field_index:
                    zones.append((vav_id, zone_temp_field_index[vav_field], self.zone_action_indices.get(vav_id, {})))

            recirc_idx = self.recirculation_indices[i] if i < len(self.recirculation_indices) else None
            oat_field = f"ahs_{ahu_id}_outside_air_temperature_sensor"
            oat_idx = names.index(oat_field) if oat_field in names else None

            self.ahu_controllers.append({
                "ahu_id": ahu_id,
                "temp_idx": self.temp_indices[i],
                "pressure_idx": self.pressure_indices[i] if i < len(self.pressure_indices) else None,
                "run_discrete_idx": i,
                "recirc_idx": recirc_idx,
                "oat_idx": oat_idx,
                "zones": zones,
                "zone_obs_indices": [obs_idx for _, obs_idx, _ in zones],
            })

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        self.water_discrete_idx = self.n_discrete - 1

        self._sensor_band_indices: dict[str, tuple[int | None, int | None]] = {}
        for ctrl in self.ahu_controllers:
            for vav_id, _, _ in ctrl["zones"]:
                h_name = f"{vav_id}_zone_air_heating_temperature_setpoint_sensor"
                c_name = f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor"
                self._sensor_band_indices[vav_id] = (
                    names.index(h_name) if h_name in names else None,
                    names.index(c_name) if c_name in names else None,
                )

        self._building_oat_idx = None
        for candidate in ("outside_air_temperature_sensor", "outside_air_temperature"):
            if candidate in names:
                self._building_oat_idx = names.index(candidate)
                break
        if self._building_oat_idx is None:
            for i, n in enumerate(names):
                if n.endswith("_outside_air_temperature_sensor"):
                    self._building_oat_idx = i
                    break

    def reset(self) -> None:
        self._zone_integral: dict[str, float] = {}
        self._zone_u: dict[str, float] = {}
        self._on_steps = [0] * self.n_ahu
        self._last_on: list[np.ndarray | None] = [None] * self.n_ahu
        self._oat_ema: float | None = None
        self._oat_min: float = 350.0
        self._oat_max: float = 200.0

    def _get_hour_and_day(self, obs: np.ndarray) -> tuple[int, int]:
        sin_h, cos_h = obs[self.hod_sin_index], obs[self.hod_cos_index]
        sin_d, cos_d = obs[self.dow_sin_index], obs[self.dow_cos_index]
        ang_h = np.arctan2(sin_h, cos_h)
        if ang_h < 0: ang_h += 2 * np.pi
        hour = int(np.floor(24 * ang_h / (2 * np.pi)))
        ang_d = np.arctan2(sin_d, cos_d)
        if ang_d < 0: ang_d += 2 * np.pi
        day = int(np.floor(7 * ang_d / (2 * np.pi)))
        return hour, day

    def _denormalize_zone_temp(self, val: float) -> float:
        return float(val * self.zone_temp_std + self.zone_temp_mean)

    def _outside_kelvin(self, val: float) -> float:
        v = float(val)
        return v if v > 150.0 else v * self.oat_std + self.oat_mean

    def _zone_band(self, obs: np.ndarray, vav_id: str, low: float, high: float) -> tuple[float, float]:
        idx = self._sensor_band_indices.get(vav_id)
        if idx is None:
            return low, high
        lo_idx, hi_idx = idx
        lo = float(obs[lo_idx]) if lo_idx is not None else low
        hi = float(obs[hi_idx]) if hi_idx is not None else high
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return low, high
        return lo, hi

    def _recirculation(self, obs: np.ndarray, ctrl: dict[str, Any], u: float, low: float) -> float:
        oat_idx = ctrl.get("oat_idx")
        if u >= 0.0 or oat_idx is None:
            return 1.0

        outside = self._outside_kelvin(obs[oat_idx])
        inside = float(np.mean([self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0

        pressure_pa = (float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0)) + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA
        pressure_pa = max(pressure_pa, MIN_STATIC_PRESSURE_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0

        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        target = max(target, low)
        if inside <= target:
            return 1.0

        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = is_weekday and (self.morning_start_hour <= hour < self.evening_start_hour)
        is_prestart_window = is_weekday and (self.morning_start_hour - self.PRESTART_H <= hour < self.morning_start_hour)
        is_pi_active = is_work_hour or is_prestart_window

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

        # Tropical check (Singapore: constantly hot, never drops below 20 C)
        is_tropical = (self._oat_min > 293.15 and self._oat_ema is not None and self._oat_ema > 296.15)

        comfort_low, comfort_high = self.comfort_temp_window
        eco_low, eco_high = self.eco_temp_window

        # Night thresholds
        night_heat_threshold = eco_low - self.NIGHT_HEAT_DROP_K

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0

        # Scan all zone temperatures
        all_zone_temps: list[float] = []
        for ctrl in self.ahu_controllers:
            for _, obs_idx, _ in ctrl["zones"]:
                all_zone_temps.append(self._denormalize_zone_temp(obs[obs_idx]))
        min_bldg_temp = min(all_zone_temps) if all_zone_temps else 294.0
        max_bldg_temp = max(all_zone_temps) if all_zone_temps else 294.0

        # Workday Thermal Coasting: in final 45 mins of workday, if comfortable, idle early
        is_coasting = (
            is_work_hour
            and (self.evening_start_hour - 0.75 <= hour < self.evening_start_hour)
            and min_bldg_temp >= comfort_low + 0.3
            and max_bldg_temp <= comfort_high - 0.3
        )

        # -------------------------------------------------------------
        # 1. ZONE DAMPER & REHEAT CONTROL
        # -------------------------------------------------------------
        self._zone_u = {}
        max_reheat_demand = 0.0
        all_shortfalls: list[float] = []

        if is_pi_active and not is_coasting:
            # Per-zone PI loop targeting band centre
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

                    # PI calculation with conditional anti-windup
                    integral = self._zone_integral.get(vav_id, 0.0)
                    u_raw = self.KP * err + self.KI * integral
                    if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                        integral = float(np.clip(integral + err, -1.0 / self.KI, 1.0 / self.KI))
                    self._zone_integral[vav_id] = integral

                    u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
                    self._zone_u[vav_id] = u
                    if u < 0:
                        max_reheat_demand = max(max_reheat_demand, -u)

                    damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u)
                    damper_idx = action_indices.get("supply_air_damper_percentage_command")
                    if damper_idx is not None:
                        continuous[damper_idx] = np.float32(2.0 * damper - 1.0)

                    reheat_idx = action_indices.get("reheat_valve_percentage_command")
                    if reheat_idx is not None:
                        continuous[reheat_idx] = np.float32(2.0 * max(0.0, -u) - 1.0)
        else:
            # Unoccupied / Night:
            # Check if any zone drops below night heating threshold (e.g. Fairbanks perimeter defense)
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
                        # Cold zone at night: protect from freeze-out
                        err = night_heat_threshold - temp
                        all_shortfalls.append(err)
                        reheat_val = min(1.0, err / 2.0)
                        max_reheat_demand = max(max_reheat_demand, reheat_val)
                        if r_idx is not None:
                            continuous[r_idx] = np.float32(2.0 * reheat_val - 1.0)
                        if d_idx is not None:
                            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * reheat_val
                            continuous[d_idx] = np.float32(2.0 * damper - 1.0)
                    else:
                        if d_idx is not None: continuous[d_idx] = np.float32(-1.0)
                        if r_idx is not None: continuous[r_idx] = np.float32(-1.0)

        # -------------------------------------------------------------
        # 2. AHU AIR HANDLER CONTROL (Demand SAT + Cooling Hold)
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

                # Demand SAT: cooling off (+1.0) when zones need heat, cold when zones need cool
                if mean_u > 0.0:
                    sat_cmd = self.COLD_SAT
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
                    self._last_on[i] = np.array([sat_cmd, self.FORCED_PRESSURE, 1.0], dtype=np.float32)
                else:
                    self._on_steps[i] = 0
            else:
                # Night: check cooling hold in extreme hot climates (Phoenix)
                if (not is_pi_active and self._on_steps[i] > 0
                        and self._on_steps[i] < self.HOLD_STEPS
                        and self._last_on[i] is not None
                        and self._last_on[i][0] < 0.0
                        and max_bldg_temp > comfort_high):
                    # Hold cooling active overnight in hot climates
                    discrete[run_idx] = 1
                    continuous[temp_idx] = self._last_on[i][0]
                    if press_idx is not None: continuous[press_idx] = np.float32(self.FORCED_PRESSURE)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] += 1
                elif all_shortfalls and any(self._denormalize_zone_temp(obs[j]) < night_heat_threshold for j in ctrl["zone_obs_indices"]):
                    # Night heating needed: AHU runs at floor pressure to deliver reheat airflow
                    discrete[run_idx] = 1
                    continuous[temp_idx] = np.float32(1.0)  # Cooling off
                    if press_idx is not None: continuous[press_idx] = np.float32(-1.0)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] = 0
                else:
                    # Deep Standby: AHU completely idle
                    discrete[run_idx] = 0
                    continuous[temp_idx] = np.float32(-1.0)
                    if press_idx is not None: continuous[press_idx] = np.float32(-1.0)
                    if recirc_idx is not None: continuous[recirc_idx] = np.float32(1.0)
                    self._on_steps[i] = 0

        # -------------------------------------------------------------
        # 3. HOT WATER PLANT CONTROL (Boiler + Hydronic Pump)
        # -------------------------------------------------------------
        if is_tropical:
            # Singapore: 100% shutdown of hot water system
            if self.water_temp_idx is not None: continuous[self.water_temp_idx] = np.float32(-1.0)
            if self.water_pressure_idx is not None: continuous[self.water_pressure_idx] = np.float32(-1.0)
            if 0 <= self.water_discrete_idx < self.n_discrete: discrete[self.water_discrete_idx] = 0
        elif max_reheat_demand > 0.0 or all_shortfalls:
            # Active heating demand (day or night)
            water_frac = max_reheat_demand if max_reheat_demand > 0 else 0.4

            # Cold climate perimeter defense (Fairbanks) only if zones are falling behind
            if oat is not None and oat < 283.15 and min_bldg_temp < comfort_low - 1.0:
                oat_floor = (283.15 - oat) / 35.0
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

        # Clip bounds and check types
        continuous = np.nan_to_num(continuous, nan=0.0, posinf=1.0, neginf=-1.0)
        continuous = np.clip(continuous, -1.0, 1.0).astype(np.float32)
        discrete = np.nan_to_num(discrete, nan=0).astype(np.int32)
        discrete = np.clip(discrete, 0, 1).astype(np.int32)

        return {"continuous_action": continuous, "discrete_action": discrete}
