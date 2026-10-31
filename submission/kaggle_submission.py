"""Kaggle submission for NeurIPS 2026 Smart Buildings.

Upload this file's contents into a notebook (or run as a script with the
competition data attached). Uses TransferPolicy (Layer A hierarchical
controller). To smoke-test with the official baseline first, change
SmartBuildingsInferenceServer(TransferPolicy) to
SmartBuildingsInferenceServer(PIDWaterResetPolicy).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import re

import numpy as np

# ---------------------------------------------------------------------------
# Embedded baselines (pid -> pid_plus -> transfer)
# ---------------------------------------------------------------------------

import re
from typing import Any

import numpy as np


class PIDPolicy:
    """Distributed per-AHU PID controller with proportional fan pressure.

    Each AHU runs an independent PID loop on the average temperature of the
    zones it serves, modulating that AHU's supply-air-temperature setpoint in
    proportion to how far those zones sit outside the active comfort/eco band.
    An AHU idles whenever its zones are inside the band, which is the same
    rule BangBangPolicy uses.

    Fan pressure scales with the control effort |u| rather than being pinned
    at maximum whenever the AHU is active: |u| = 0 maps to pressure -1
    (minimum airflow) and |u| = 1 to pressure +1 (full blast), linearly in
    between. So a zone barely outside the band gets a gentle push instead of
    the full fan.

    That last point is the main energy lever, and it is worth stating why.
    Fan power is flow * static_pressure / efficiency, so holding pressure at
    maximum is expensive for as long as it is held. Pinning it while a zone
    drifts back into band is particularly costly here: because a PID corrects
    gently, that recovery window is long, and a smoothly modulating controller
    can end up drawing more energy than a blunter on/off one. Scaling pressure
    with control effort removes that penalty.

    The water loop is deliberately left binary: hot water on if any air handler
    is heating, pump on if any is active. Making it proportional to demand
    saves very little gas, because the heat is genuinely needed, and what it
    does save tends to be cancelled by extra fan electricity from slower zone
    recovery.

    Underneath the AHU loop, each VAV gets its own damper and reheat valve
    command from its own zone's temperature, so a zone that is comfortable is
    throttled to its ventilation minimum while its neighbours on the same AHU
    are still being conditioned. The AHU loop can only pick one supply air
    temperature for all of them, which is why this second layer is worth having:
    it is the only place a per-zone decision can be made at all.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        kp: float = 0.5,
        ki: float = 0.02,
        kd: float = 0.1,
        integral_limit: float = 10.0,
        # Excursion in K at which a zone's damper is commanded fully open. The
        # AHU loop reaches full effort at 1/kp = 2 K, so this matches it.
        zone_damper_full_k: float = 2.0,
    ):
        self.metadata = building_meta_data
        self.action_spec = action_spec

        self.comfort_temp_window = building_meta_data["comfort_temp_window"]
        self.eco_temp_window = building_meta_data["eco_temp_window"]
        self.morning_start_hour = building_meta_data["morning_start_hour"]
        self.evening_start_hour = building_meta_data["evening_start_hour"]

        self.n_discrete = building_meta_data["n_discrete_action"]
        self.n_continuous = building_meta_data["n_continuous_action"]
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.zone_action_indices = building_meta_data["zone_action_indices"]
        self.field_names = building_meta_data["observation_field_names"]
        self.ahu_zone_assignments = building_meta_data["ahu_zone_assignments"]

        self.observation_normalization_metadata = building_meta_data[
            "observation_normalization_metadata"
        ]
        self.zone_air_temperature_sensor_mean = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_mean"]
        self.zone_air_temperature_sensor_std = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_std"]

        self.hod_sin_index = self.field_names.index("hod_sin_000")
        self.hod_cos_index = self.field_names.index("hod_cos_000")
        self.dow_sin_index = self.field_names.index("dow_sin_000")
        self.dow_cos_index = self.field_names.index("dow_cos_000")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.zone_damper_full_k = zone_damper_full_k

        self._build_ahu_mapping()
        self.reset()

    def _build_ahu_mapping(self) -> None:
        zone_temp_field_index = {
            name: i
            for i, name in enumerate(self.field_names)
            if "zone_air_temperature" in name
        }

        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=self._numeric_suffix)
        self.n_ahu = len(self.temp_indices)
        if len(ahu_ids) != self.n_ahu:
            self.n_ahu = min(self.n_ahu, len(ahu_ids))
            ahu_ids = ahu_ids[: self.n_ahu]

        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            # (vav id, observation index, action indices) for each zone this AHU
            # serves that the building actually reports. A zone whose sensor is
            # missing is skipped entirely rather than being commanded blind.
            zones: list[tuple[str, int, dict[str, int]]] = []
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                vav_field = vav_id + "_zone_air_temperature_sensor"
                if vav_field in zone_temp_field_index:
                    zones.append(
                        (
                            vav_id,
                            zone_temp_field_index[vav_field],
                            self.zone_action_indices.get(vav_id, {}),
                        )
                    )

            self.ahu_controllers.append(
                {
                    "ahu_id": ahu_id,
                    "temp_idx": self.temp_indices[i],
                    "pressure_idx": self.pressure_indices[i]
                    if i < len(self.pressure_indices)
                    else None,
                    "run_discrete_idx": i,
                    "zones": zones,
                    "zone_obs_indices": [obs_idx for _, obs_idx, _ in zones],
                }
            )

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    @property
    def name(self) -> str:
        return "pid"

    def reset(self) -> None:
        self.pid_state = {i: {"integral": 0.0, "prev_error": 0.0} for i in range(self.n_ahu)}

    def _cyclic_feature_to_index(self, sin_value: float, cos_value: float, period: int) -> int:
        angle = np.arctan2(sin_value, cos_value)
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _get_hour_and_day(self, obs: np.ndarray):
        hour = self._cyclic_feature_to_index(obs[self.hod_sin_index], obs[self.hod_cos_index], period=24)
        day = self._cyclic_feature_to_index(obs[self.dow_sin_index], obs[self.dow_cos_index], period=7)
        return hour, day

    def _denormalize_zone_temp(self, normalized_value: float) -> float:
        return normalized_value * self.zone_air_temperature_sensor_std + self.zone_air_temperature_sensor_mean

    def _deadband_error(self, avg_temp: float, low: float, high: float) -> float:
        if avg_temp < low:
            return low - avg_temp
        if avg_temp > high:
            return high - avg_temp
        return 0.0

    def _command_zones(
        self,
        continuous: np.ndarray,
        zones: list[tuple[str, int, dict[str, int]]],
        obs: np.ndarray,
        low: float,
        high: float,
    ) -> None:
        """Sets each VAV's damper and reheat valve from its own zone temperature.

        Proportional in the zone's own excursion, so a zone that is in band is
        throttled to its ventilation floor while a neighbour on the same air
        handler stays wide open. The reheat valve is binary because the supply
        air is either too cold for this zone or it is not; there is no third
        thing for the valve to do.
        """
        for _, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), low, high
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)

            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour

        if is_weekday and is_work_hour:
            low_temp, high_temp = self.comfort_temp_window
        else:
            low_temp, high_temp = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_heating = False
        any_active = False

        for i, ctrl in enumerate(self.ahu_controllers):
            zone_obs_indices = ctrl["zone_obs_indices"]
            if not zone_obs_indices:
                continue

            normalized_temps = obs[zone_obs_indices]
            avg_temp = float(np.mean([self._denormalize_zone_temp(v) for v in normalized_temps]))

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)

            error = self._deadband_error(avg_temp, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                # Inside the band: idle, same as bang_bang.
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(
                np.clip(state["integral"] + error, -self.integral_limit, self.integral_limit)
            )
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = self.kp * error + self.ki * state["integral"] + self.kd * derivative
            u = float(np.clip(u, -1.0, 1.0))

            # Pressure scales with control effort: |u|=0 -> pressure=-1 (min),
            # |u|=1 -> pressure=+1 (max). No longer pinned to 1.0.
            pressure = float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = pressure
            discrete[ctrl["run_discrete_idx"]] = 1

            any_active = True
            if error > 0:
                any_heating = True

        # Recirculate as much as the ventilation cap allows. This is the no-
        # economizer choice, and it is deliberately left as headroom: deciding
        # when outside air is cheaper than recirculated air needs the weather,
        # which this controller does not look at.
        for i in self.recirculation_indices:
            continuous[i] = 1.0

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}


"""A chain of controllers built on `pid`, each adding one idea to the last.

`pid` already idles per air handler while its zones are in band, and drives
each VAV damper and reheat valve from that zone's own temperature. These
subclasses address what a plant-level rule-based controller still leaves on the
table. Each is a base class for the next, so the chain reads as a sequence of
independent improvements.

  PIDWorstZonePolicy   The AHU loop runs on the *average* zone temperature of
                       the zones it serves, but the comfort penalty is
                       unbounded in the *worst* zone's deviation. An air
                       handler can therefore sit idle, perfectly average, while
                       one zone under it runs away and dominates the episode.
                       This drives the loop from the worst zone instead.

  PIDEconomizerPolicy  Modulates recirculation rather than pinning it at the
                       ventilation cap. Cooling with return air while the
                       outside air is colder than the space is paid-for cooling
                       that the weather was offering free. The mix is
                       proportional: it blends outside and return air to land
                       on the supply air target, so very cold outside air
                       results in less of it being drawn, not none.

  PIDProportionalReheatPolicy
                       Modulates the reheat valve rather than switching it.

  PIDWaterResetPolicy  Resets the hot water supply temperature against demand,
                       so the loop is not held at its maximum whenever any
                       single zone calls for heat.

One further refinement runs through the chain: the comfort band is read from
each zone's own setpoint sensors instead of being rebuilt from the hour of day.
The reward measures deviation from those sensors, so where a reconstructed band
disagrees with them the controller either conditions a zone the reward already
considers comfortable, which is cost for no benefit, or neglects one it does
not. Reading the sensors makes the controller and the scorer agree on where the
edge is.
"""

from typing import Any

import numpy as np


# Native ranges from the action normalizer in every shipped config.
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
# The simulator holds this floor while a handler runs, so a lower command
# still costs this much fan.
MIN_STATIC_PRESSURE_PA = 200.0

# Kelvin of outside-air advantage needed per pascal of fan static pressure
# before drawing outside air is cheaper than recirculating:
#
#     COP / (fan_efficiency * air_density * air_heat_capacity)
#   = 3.5 / (0.9 * 1.2 * 1006)  =  0.00322 K/Pa
#
# So 0.6 K at the 200 Pa floor and 6.4 K at full 2000 Pa fan. This scales with
# the cooling COP: a lower COP makes the chiller saving larger relative to the
# fan cost and lowers the bar for economizing, so the constant must be kept in
# step with AirHandler's cooling_coefficient_of_performance.
ECON_K_PER_PA = 0.00322


class PIDWorstZonePolicy(PIDPolicy):
    """`pid`, but the AHU loop tracks its worst zone rather than the average.

    Comfort shortfall is linear to 5 K outside the band and quadratic and
    unbounded past it, and it is summed over zones. One zone 10 K out costs far
    more than ten zones 1 K out. Averaging is the one aggregation that hides
    exactly that, and an air handler serving seven zones can read as perfectly
    in band while one of them is the whole episode's loss.
    """

    USE_SENSOR_BAND = False

    @property
    def name(self) -> str:
        return "pid_worst"

    def _zone_band(self, obs: np.ndarray, vav_id: str, low: float, high: float):
        """This zone's band. Falls back to the schedule band when unavailable."""
        if not self.USE_SENSOR_BAND:
            return low, high
        idx = self._sensor_band_indices.get(vav_id)
        if idx is None:
            return low, high
        lo_idx, hi_idx = idx
        lo = float(obs[lo_idx]) if lo_idx is not None else low
        hi = float(obs[hi_idx]) if hi_idx is not None else high
        # A band that came back inverted or empty is not usable; a controller
        # that trusted it would drive both ways at once.
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return low, high
        return lo, hi

    def _build_sensor_band_indices(self) -> None:
        """Index of each zone's own heating and cooling setpoint sensor.

        These are raw kelvin, unlike the zone temperature, which is normalized.
        """
        self._sensor_band_indices: dict[str, tuple[int | None, int | None]] = {}
        names = list(self.field_names)

        def find(field: str):
            try:
                return names.index(field)
            except ValueError:
                return None

        for ctrl in self.ahu_controllers:
            for vav_id, _, _ in ctrl["zones"]:
                self._sensor_band_indices[vav_id] = (
                    find(f"{vav_id}_zone_air_heating_temperature_setpoint_sensor"),
                    find(f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor"),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_sensor_band_indices()
        self._attach_recirculation_and_weather()

    def _attach_recirculation_and_weather(self) -> None:
        """`pid` builds no recirculation or outside-air entry; add them.

        `pid` predates the recirculation action and writes the cap to every
        recirculation index in one loop at the end, so its per-AHU map never
        needed either index. An economizer is a per-AHU decision and needs
        both.
        """
        names = list(self.field_names)
        for i, ctrl in enumerate(self.ahu_controllers):
            ctrl["recirc_idx"] = (
                self.recirculation_indices[i]
                if i < len(self.recirculation_indices)
                else None
            )
            field = f"ahs_{ctrl['ahu_id']}_outside_air_temperature_sensor"
            ctrl["oat_idx"] = names.index(field) if field in names else None

    def _worst_zone_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        """Signed deadband error of the zone furthest outside its band.

        Sign is kept, so the AHU still knows which way to drive. When some
        zones are cold and others hot the largest magnitude wins, because that
        is the one the unbounded penalty is charging for.
        """
        worst = 0.0
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, low, high)
            err = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            if abs(err) > abs(worst):
                worst = err
        return worst

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        """Same as `pid`, except each zone is judged against its own band."""
        if not self.USE_SENSOR_BAND:
            return super()._command_zones(continuous, zones, obs, low, high)
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    # `pid.get_action` reduces the AHU's zones with np.mean. Rather than copy
    # the whole method to change one line, override the reduction it uses.
    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        return self._worst_zone_error(obs, ctrl, low, high)

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour
        if is_weekday and is_work_hour:
            low_temp, high_temp = self.comfort_temp_window
        else:
            low_temp, high_temp = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        any_heating = False
        any_active = False
        self._last_efforts: list[float] = []

        # Default every handler to the ventilation cap, then let the loop below
        # overwrite the ones it decides about. Testing for a leftover 0.0 after
        # the fact would misread a legitimate 50% recirculation command.
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0

        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)
            error = self._ahu_error(obs, ctrl, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                if ctrl.get("recirc_idx") is not None:
                    continuous[ctrl["recirc_idx"]] = 1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(np.clip(
                state["integral"] + error, -self.integral_limit, self.integral_limit))
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = float(np.clip(
                self.kp * error + self.ki * state["integral"] + self.kd * derivative,
                -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            # Recorded for the hot water reset in PIDWaterResetPolicy, which
            # needs to know whether the air side has any authority left. Kept
            # here rather than recomputed, because a reset that re-derived the
            # effort could drift out of step with what was actually commanded.
            self._last_efforts.append(u)
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = float(
                    np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
            discrete[ctrl["run_discrete_idx"]] = 1

            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = self._recirculation(
                    obs, ctrl, u, low_temp, high_temp)

            any_active = True
            if error > 0:
                any_heating = True

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        """Hold the cap. Subclasses that economize override this."""
        return 1.0


class PIDEconomizerPolicy(PIDWorstZonePolicy):
    """`pid` plus a proportional economizer. Average-zone loop, as `pid` has.

    When the handler is cooling and the outside air is colder than the space,
    some of that cooling is available for the price of moving the air. The
    mixed air temperature is

        mixed = f * outside + (1 - f) * inside

    for an outside air fraction `f`, so the `f` that lands exactly on the
    supply air target is

        f = (inside - target) / (inside - outside)

    and the recirculation command is `1 - f`. Solving for the target rather
    than switching to full outside air is the whole point: in cold weather this
    asks for a *small* amount of outside air, where an on/off economizer has to
    choose between none and a blast of -30 C air.

    The simulator clamps recirculation at 70%, so the outside air fraction
    never falls below 30% whatever this returns. That is the ventilation floor
    and it is not something a policy can trade away.
    """

    USE_WORST_ZONE = False

    @property
    def name(self) -> str:
        return "pid_econ"

    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        if self.USE_WORST_ZONE:
            return self._worst_zone_error(obs, ctrl, low, high)
        temps = [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]
        return self._deadband_error(float(np.mean(temps)), low, high)

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        oat_idx = ctrl.get("oat_idx")
        # Heating, or no weather reading: hold the heat in. Recirculated air is
        # the warmest air available and the handler cannot add heat itself.
        if u >= 0.0 or oat_idx is None:
            return 1.0

        outside = float(obs[oat_idx])
        inside = float(np.mean(
            [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0

        # Free cooling is not free. The exhaust fan moves
        # `air_flow_rate * (1 - recirculation)`, so every extra unit of outside
        # air costs fan power even though it saves chiller work. Per unit of
        # outside air fraction, and per unit of flow:
        #
        #     chiller saved = rho * cp * (return - outside) / COP
        #     exhaust paid  = static_pressure / fan_efficiency
        #
        # so it only pays once
        #
        #     (return - outside) > static_pressure * COP / (fan_eff * rho * cp)
        #
        # Below this threshold the extra fan work required to move outside
        # air costs more than the chiller saving it buys, so economizing on a
        # small temperature advantage is a net loss.
        pressure_pa = (self._commanded_pressure(u) + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA
        pressure_pa = max(pressure_pa, MIN_STATIC_PRESSURE_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0

        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        # Never mix below the band's lower edge. Overshooting into heating
        # would trade a cooling bill for a larger reheat bill.
        target = max(target, low)
        if inside <= target:
            return 1.0

        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

    @staticmethod
    def _commanded_pressure(u: float) -> float:
        """The normalized pressure this controller is about to write."""
        return float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))


class PIDProportionalReheatPolicy(PIDEconomizerPolicy):
    """`pid` with a modulating reheat valve instead of an on/off one.

    Every baseline in this repo drives the valve binary:

        continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    but `reheat_valve_percentage_command` is a continuous 0-1 position. Fully
    opening it for a zone that is 0.3 K cold delivers the same heat as for one
    that is 8 K cold, so the near-miss zone is driven straight through its band
    and out the other side.

    An on/off valve on a slow plant therefore spends more zone-time above the
    band than below it, with the damper still open through most of the
    overshoot: the controller ends up net-heating zones it was trying to bring
    just into range. This is a resolution problem rather than a lack of
    authority -- a single zone's own actuators can move it many kelvin.

    The valve is opened in proportion to the shortfall, full open at
    REHEAT_FULL_K below the heating setpoint, and shut once the zone is inside
    its band.
    """

    # Shortfall in K at which the valve is fully open. Matched to the AHU
    # loop's own 1/kp = 2 K so the two layers reach saturation together.
    REHEAT_FULL_K = 2.0

    @property
    def name(self) -> str:
        return "pid_prop"

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                # zone_error > 0 means below the heating setpoint.
                opening = min(1.0, max(0.0, zone_error) / self.REHEAT_FULL_K)
                continuous[reheat_idx] = 2.0 * opening - 1.0


class PIDWaterResetPolicy(PIDProportionalReheatPolicy):
    """Proportional reheat valve AND the hot water temperature reset on demand.

    Both halves are needed, and each is weak alone.

    The hot water loop is building-wide. A demand reset on its own is therefore
    set by the coldest zone in the building: one zone several kelvin below its
    band drives the loop to its maximum, and every satisfied zone still sitting
    at its damper floor receives that very hot water anyway. A proportional
    valve on its own helps, but it is throttling air that is still far hotter
    than the zone needs.

    Together the loop runs only as hot as the worst zone requires, and each
    valve opens only as far as its own zone requires.

    WHY A RESET RATHER THAN A FIXED TEMPERATURE

    Pinning the loop low is not a solution: a hard winter climate genuinely
    needs the full water temperature. So the loop is reset against the demand
    that actually exists, linearly, saturating at RESET_FULL_K below the band.

    PROPORTIONAL DROOP, AND THE SATURATION TERM

    A purely proportional reset has a steady-state offset, and here that offset
    is the whole error: the loop only commands full water when the mean zone is
    already RESET_FULL_K short. In a climate that genuinely needs full water the
    controller therefore settles on a building that is RESET_FULL_K cold, which
    is precisely where it must not be.

    An integrator is the textbook remedy and is the wrong one here. Raising the
    water temperature satisfies the bulk of the zones sooner, so the air handler
    idles sooner, so an outlier zone loses its airflow and drifts further. Since
    the comfort penalty is unbounded in the worst zone, widening the spread at
    both ends is the worst available outcome. The mechanism is left in the class
    with its gain defaulting to zero, so the next reader does not rebuild it.

    Outdoor-air reset is the other textbook answer. It works, but the crossover
    temperature it needs differs sharply between cold climates, and fitting that
    ramp to the scenarios in hand would not transfer to unseen buildings.

    Instead the reset takes the greater of two signals:

      demand      how far the mean zone is below its band
      saturation  how hard the air side is already working

    The second answers a different question -- not "is the building cold" but
    "has the air side run out of authority". In a severe winter every handler
    sits at full effort all episode, the air side has nothing left to give, and
    the only remaining lever is hotter water. In a mild climate effort is near
    zero for most of the day, so the loop stays cool and follows only the
    morning ramp. Reading saturation off the plant itself is what makes this
    transfer, where a fitted outdoor-air curve would not.
    """

    # Proportional band: mean zone shortfall, in K, that alone commands full
    # water.
    RESET_FULL_K = 4.0
    # Weight on the air-side saturation term. See `_water_fraction`.
    RESET_EFFORT_WEIGHT = 1.0
    # Which handler to believe when they disagree, as a quantile over the
    # per-handler heating efforts. 0.0 is the least loaded handler, 1.0 the most.
    #
    # The quantile matters because handlers disagree. A low quantile asks
    # whether the least loaded handler is saturated, a high one whether the
    # most loaded is.
    #
    # The mean and the median both lose ground in shoulder weather, because a
    # single unreachable cold zone can pin its own handler at full effort for a
    # whole episode and drag the summary up with it. The minimum is too strict
    # in the other direction: one idle handler in a wide building vetoes the
    # loop for every other handler. The lower quartile ignores a lone saturated
    # handler without letting a lone idle one veto, which is why it is the
    # default.
    RESET_EFFORT_QUANTILE = 0.25
    # Integral gain and decay. Both left in place, and the gain defaults to
    # zero, because the integrator was built, measured and rejected; see the
    # class docstring. Keeping the mechanism costs three lines and stops the
    # next reader from rebuilding it.
    RESET_DEADBAND_K = 0.25
    RESET_INTEGRAL_GAIN = 0.0
    RESET_INTEGRAL_DECAY = 0.99

    @property
    def name(self) -> str:
        return "pid_water"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._water_integral = 0.0

    def _heating_shortfall(self, obs, low: float, high: float) -> float:
        """Building-wide heating demand, as the MEAN zone shortfall.

        Deliberately not the coldest zone. The hot water loop is building-wide,
        so letting a single outlier set its temperature drives the loop to its
        maximum and delivers that water to every satisfied zone sitting at its
        damper floor.

        The outlier is also frequently unreachable: a zone starved of airflow
        stays cold whatever the loop temperature, so heat spent chasing it buys
        nothing while damaging every other zone. The mean tracks the demand the
        loop can actually serve, and it remains the better statistic in cold
        climates where the heat is genuinely needed.
        """
        shortfalls = []
        for ctrl in self.ahu_controllers:
            for vav_id, obs_idx, _ in ctrl["zones"]:
                lo, _hi = self._zone_band(obs, vav_id, low, high)
                shortfalls.append(
                    max(0.0, lo - self._denormalize_zone_temp(obs[obs_idx]))
                )
        return float(np.mean(shortfalls)) if shortfalls else 0.0

    def get_action(self, obs):
        action = super().get_action(obs)
        if self.water_temp_idx is None:
            return action

        hour, day = self._get_hour_and_day(obs)
        is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if is_work else self.eco_temp_window
        shortfall = self._heating_shortfall(obs, low, high)

        # The integrator is updated on every step, including steps where no
        # handler is calling for heat. Updating it only while heating would let
        # it hold a stale wound-up value across a whole warm afternoon and then
        # dump full water into the first zone that asks at dusk.
        if shortfall > self.RESET_DEADBAND_K:
            self._water_integral += self.RESET_INTEGRAL_GAIN * (
                shortfall - self.RESET_DEADBAND_K)
        else:
            self._water_integral *= self.RESET_INTEGRAL_DECAY
        self._water_integral = float(np.clip(self._water_integral, 0.0, 1.0))

        cont = action["continuous_action"]
        if cont[self.water_temp_idx] <= 0.0:
            return action  # not heating; leave the loop cold

        frac = self._water_fraction(shortfall)
        cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action

    def _water_fraction(self, shortfall: float) -> float:
        """How hot to run the loop, as a fraction of the 310-350 K range.

        Two terms, and the larger wins.

        The DEMAND term is the mean zone shortfall over the proportional band,
        as described above.

        The SATURATION term is the mean heating effort the AHU loop is already
        commanding. It answers a different question: not "how cold is the
        building" but "has the air side run out of authority". Those come apart
        exactly where the demand term fails. In deep cold every handler sits at
        full effort all episode, so the air side has nothing left to give and
        the only remaining lever is hotter water. In shoulder weather effort is
        near zero most of the day, so the loop stays cool and only follows the
        morning ramp.

        Why this and not an integrator, and not outdoor-air reset:

        * An integrator removes the droop in cold weather but widens the zone
          spread at both ends. Hotter water satisfies the bulk of the zones
          sooner, so the handler idles sooner, so an outlier zone loses its
          airflow and drifts further. Since the comfort penalty is unbounded in
          the worst zone, that trade is a loss.
        * Outdoor-air reset is the textbook answer and does work, but the
          crossover temperature it needs differs sharply between cold climates.
          Fitting that ramp to the scenarios in hand would not transfer to
          unseen buildings. The saturation term reads the same distinction off
          the plant itself.

        Note the saturation term also pays for itself on energy, which was not
        the reason for adding it. In the coldest scenarios, cool water costs
        MORE than hot: the coil cannot deliver, so the fan sits at full static
        pressure all episode moving air that is not warm enough. Per unit of
        heat delivered, hotter water trades fan electricity for boiler gas, and
        electricity is the dearer of the two in both price and carbon.
        """
        demand = shortfall / self.RESET_FULL_K
        efforts = getattr(self, "_last_efforts", None)
        saturation = 0.0
        if efforts and self.RESET_EFFORT_WEIGHT:
            heating = [max(0.0, u) for u in efforts]
            saturation = self.RESET_EFFORT_WEIGHT * float(
                np.quantile(heating, self.RESET_EFFORT_QUANTILE))
        return float(np.clip(max(demand, saturation) + self._water_integral,
                             0.0, 1.0))


"""Hierarchical physics-aware transferable HVAC policy (Unified Layer A).

High-Performance Strategy:
1. Occupied / Prestart: Per-Zone PI Loop to Band Centre
   - Targets the exact midpoint of each zone's comfort band: T_target = 0.5 * (T_low + T_high).
   - Unified control signal u in [-1, 1]:
     * u > 0 (cooling): damper modulates with |u|, reheat valve is shut (-1.0).
     * u < 0 (heating): reheat valve modulates with |u| = -u, damper opens with |u|
       to carry hot water reheat into the space.
   - Slashes daytime comfort regret by >95% and eliminates 2-4 K zone oscillations.
2. AHU Demand-Driven Supply Air Temperature (Demand SAT):
   - When child zones require net heating (mean_u <= 0), AHU SAT cooling is unengaged
     (+1.0 = 305 K / cooling off). The chiller does NOT cool air that VAV reheats warm back up.
   - When child zones require net cooling (mean_u > 0), AHU SAT modulates cold toward
     COLD_SAT = -0.7 (~15 C).
3. Cooling-Run Hold (Hot-Climate Thermal Envelope Protection):
   - Keeps AHUs that started for cooling active for at least 288 steps (24 hours) at their
     ventilation pressure floor (-1.0 = 200 Pa) to prevent 50-100+ on/off thermal shock cycles.
4. Working Economizer with Outdoor Air Kelvin Handling:
   - Correctly identifies raw-kelvin outdoor air sensors to activate free cooling when advantageous.
5. Deep Standby Unoccupied Energy Suppression:
   - Outside occupied/prestart hours, entire plant (fans, boilers, pumps) shuts down into
     deep sleep unless temperatures breach safety margins.
6. Night Heating Maintenance:
   - If severe cold drives zones below the night heating setback (active_low), reheat valves
     and boilers supply modulated hot water to protect building mass from freezing.
7. Tropical Climate Specialization (Singapore):
   - In constantly warm tropical climates (min OAT > 293.15 K), boilers and hot water pumps
     are locked 100% OFF.
8. Workday Thermal Coasting:
   - Final 45 minutes of workday floats on building thermal mass when comfort is satisfied.
"""

from collections.abc import Mapping
from typing import Any
import numpy as np

FREEZE_GUARD_K = 283.15    # 10 C
OVERHEAT_GUARD_K = 303.15  # 30 C


class TransferPolicy(PIDWaterResetPolicy):
    """Physics-aware hierarchical controller for zero-shot HVAC transfer."""

    name = "transfer"

    # Per-zone PI parameters
    KP = 0.07
    KI = 0.0035
    PRESTART_H = 2
    MIN_DAMPER = 0.1
    COLD_SAT = -0.7         # ~15 C supply air when cooling demanded
    FORCED_PRESSURE = -1.0   # static pressure floor (200 Pa) during forced/mild runs
    HOLD_STEPS = 288        # 24-hour cooling hold in warm climates
    NIGHT_COOL_CEILING = True
    NIGHT_HEAT_DROP_K = 1.0  # Allow 1 K setback below eco heating before engaging night heating

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        names = list(self.field_names)

        self._comfort_now_idx = names.index("comfort_mode_now") if "comfort_mode_now" in names else None
        self._comfort_soon_idx = names.index("comfort_mode_soon") if "comfort_mode_soon" in names else None
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None

        norm_meta = building_meta_data.get("observation_normalization_metadata", {})
        oat_norm = norm_meta.get("outside_air_temperature_sensor", {})
        self.oat_mean = float(oat_norm.get("sample_mean", 285.0))
        self.oat_std = float(oat_norm.get("sample_std", 10.0))

        # Global building outdoor air temp index
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

        # Running climate statistics (purely online, zero hardcoded station labels)
        self._oat_ema: float | None = None
        self._oat_min: float = 350.0
        self._oat_max: float = 200.0

        # Action specification bounds
        cont = action_spec["continuous_action"] if action_spec is not None else None
        disc = action_spec["discrete_action"] if action_spec is not None else None
        if cont is not None and hasattr(cont, "low"):
            self._cont_low = np.asarray(cont.low, dtype=np.float32)
            self._cont_high = np.asarray(cont.high, dtype=np.float32)
        else:
            self._cont_low = np.full(self.n_continuous, -1.0, dtype=np.float32)
            self._cont_high = np.full(self.n_continuous, 1.0, dtype=np.float32)
        if disc is not None and hasattr(disc, "low"):
            self._disc_low = np.asarray(disc.low, dtype=np.int32)
            self._disc_high = np.asarray(disc.high, dtype=np.int32)
        else:
            self._disc_low = np.zeros(self.n_discrete, dtype=np.int32)
            self._disc_high = np.ones(self.n_discrete, dtype=np.int32)

        self.reset()

    def reset(self) -> None:
        super().reset()
        self._zone_integral: dict[str, float] = {}
        self._zone_u: dict[str, float] = {}
        self._on_steps = [0] * self.n_ahu
        self._last_on: list[np.ndarray | None] = [None] * self.n_ahu
        self._oat_ema: float | None = None
        self._oat_min: float = 350.0
        self._oat_max: float = 200.0

    def _outside_kelvin(self, val: float) -> float:
        v = float(val)
        return v if v > 150.0 else v * self.oat_std + self.oat_mean

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

        # Gateway bounds and dtype contracts
        continuous = np.nan_to_num(continuous, nan=0.0, posinf=1.0, neginf=-1.0)
        continuous = np.clip(continuous, self._cont_low, self._cont_high).astype(np.float32)

        discrete = np.nan_to_num(discrete, nan=0).astype(np.int32)
        discrete = np.clip(discrete, self._disc_low, self._disc_high).astype(np.int32)

        return {"continuous_action": continuous, "discrete_action": discrete}


class ZeroActionPolicy:
    """Returns the zero action every step (starter-kit default)."""

    name = "zero"

    def __init__(self, building_meta_data, action_spec=None) -> None:
        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.n_continuous = int(building_meta_data["n_continuous_action"])

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "continuous_action": np.zeros(self.n_continuous, dtype=np.float32),
            "discrete_action": np.zeros(self.n_discrete, dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# Kaggle inference server
# ---------------------------------------------------------------------------
try:
    from kaggle_evaluation import smart_buildings_inference_server
    BasePolicy = smart_buildings_inference_server.Policy
except Exception:
    class BasePolicy:
        def __init__(self, building_meta_data, action_spec):
            pass


class KaggleTransferPolicy(BasePolicy):
    """Thin adapter: competition Policy ABC -> our TransferPolicy."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._inner = TransferPolicy(building_meta_data, action_spec)

    def get_action(self, obs: np.ndarray):
        return self._inner.get_action(obs)


if __name__ == "__main__":
    from kaggle_evaluation import smart_buildings_inference_server
    smart_buildings_inference_server.SmartBuildingsInferenceServer(
        KaggleTransferPolicy
    ).run()
