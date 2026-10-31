# ===== CELL 0 =====
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CANDIDATES = [Path(os.environ["SMART_BUILDINGS_DATA_ROOT"])] if os.getenv("SMART_BUILDINGS_DATA_ROOT") else []
CANDIDATES += [
    Path("/kaggle/input/competitions/neurips-2026-smart-buildings"),
    Path("/kaggle/input/neurips-2026-smart-buildings"),
]
DATA_ROOT = next((p for p in CANDIDATES if (p / "smart_buildings").is_dir()), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Competition data not found. Attach it to the notebook or set SMART_BUILDINGS_DATA_ROOT. Checked: %s" % CANDIDATES
    )
for sub in (DATA_ROOT, DATA_ROOT / "kaggle_evaluation"):
    if not sub.is_dir():
        raise FileNotFoundError("Missing folder: %s" % sub)
    sys.path.insert(0, str(sub))

IS_RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None
print("Data root:", DATA_ROOT, "| hidden rerun:", IS_RERUN)

# ===== CELL 2 =====
"""EDA of released training data; never opens evaluation_scenarios."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def analyze_data(data_root, output_root):
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
    previous = Path.cwd()
    buildings, weather_rows = [], []
    frames = {}
    try:
        os.chdir(data_root)
        scenario_dirs = sorted((data_root / "smart_buildings/training_scenarios").glob("*/*"))
        seen = set()
        for scenario in scenario_dirs:
            if not (scenario / "config.gin").exists():
                continue
            cfg = yaml.safe_load((scenario / "config.yaml").read_text())
            if scenario.parent.name not in seen:
                seen.add(scenario.parent.name)
                env = load_hybrid_environment(scenario / "config.gin")
                # Competition input is read-only on Kaggle, including metric paths.
                env.metrics_path = None
                meta = create_building_meta_data(env, scenario)
                obs, _ = env.reset()
                assert np.isfinite(obs).all(), "Non-finite initial observation"
                names = meta["observation_field_names"]
                zone_fields = [n for n in names if n.endswith("_zone_air_temperature_sensor")]
                v, k = len(meta["zone_action_indices"]), len(meta["temp_indices"])
                assert meta["n_continuous_action"] == 3*k + 2 + 2*v
                assert len(zone_fields) == v
                floor = np.load(scenario / "floor_plan.npy")
                buildings.append(dict(
                    building=scenario.parent.name, floors=meta["num_floors"],
                    zones=v, ahus=k, observation_dim=len(names),
                    continuous_dim=meta["n_continuous_action"],
                    discrete_dim=meta["n_discrete_action"],
                    grid_cells=int(floor.size), initial_missing=int(np.isnan(obs).sum()),
                    exterior_conductivity=cfg["materials"]["exterior_cv_conductivity"],
                    exterior_density=cfg["materials"]["exterior_cv_density"],
                    interior_mass_density=cfg["materials"]["interior_mass_cv_density"],
                ))
                if len(buildings) == 1:
                    pd.DataFrame(dict(field=names, initial_observation=obs)).to_csv(
                        output_root / "observation_example.csv", index=False)
                    for key in ("observation_normalization_metadata", "action_normalization_metadata"):
                        import json
                        (output_root / f"{key}.json").write_text(json.dumps(meta[key], indent=2))
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.imshow(floor, cmap="viridis", interpolation="nearest")
                    ax.set(title=f"{scenario.parent.name}: simulator floor grid", xlabel="grid column", ylabel="grid row")
                    fig.tight_layout(); fig.savefig(output_root / "floor_plan.png", dpi=150); plt.close(fig)
            csv = data_root / "smart_buildings/weather" / cfg["weather"]["weather_csv"]
            if csv not in frames:
                frame = pd.read_csv(csv)
                frame["timestamp"] = pd.to_datetime(frame["Time"], utc=True)
                frames[csv] = frame
            frame = frames[csv]
            # Gin's actual simulator clock takes the weather override.
            start = pd.Timestamp(cfg["weather"]["start_timestamp"], tz="UTC")
            days = int(cfg["simulation"]["num_days_in_episode"])
            window = frame[(frame.timestamp >= start) & (frame.timestamp < start + pd.Timedelta(days=days))]
            temp = (window.TempF - 32) * (5/9)
            weather_rows.append(dict(
                building=scenario.parent.name, weather=scenario.name,
                station=scenario.name.rsplit("_", 1)[0], start=str(start),
                hours=len(window), missing_temperatures=int(temp.isna().sum()),
                duplicate_timestamps=int(window.timestamp.duplicated().sum()),
                min_C=float(temp.min()), mean_C=float(temp.mean()), max_C=float(temp.max()),
                hours_below_minus6_8_pct=float(100*(temp < -6.8).mean()),
            ))
    finally:
        os.chdir(previous)
    building_df, weather_df = pd.DataFrame(buildings), pd.DataFrame(weather_rows)
    building_df.to_csv(output_root / "buildings.csv", index=False)
    weather_df.to_csv(output_root / "weather_strips.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    building_df.plot.bar(x="building", y=["zones", "ahus"], ax=axes[0], rot=25)
    axes[0].set_title("Variable building and action size")
    grouped = weather_df.groupby("station").agg(min_C=("min_C", "min"), mean_C=("mean_C", "mean"), max_C=("max_C", "max"))
    x = np.arange(len(grouped))
    axes[1].vlines(x, grouped.min_C, grouped.max_C, linewidth=5, color="#3b82f6")
    axes[1].scatter(x, grouped.mean_C, color="#f97316", label="mean across strips")
    axes[1].set(xticks=x, xticklabels=grouped.index, ylabel="outside temperature (C)", title="30-day training weather windows")
    axes[1].legend(); fig.tight_layout(); fig.savefig(output_root / "eda.png", dpi=150); plt.close(fig)
    print(building_df.to_string(index=False))
    print(grouped.round(2).to_string())
    print(f"Scenarios: {len(weather_df)}, initial observation NaNs: {building_df.initial_missing.sum()}")
    return building_df, weather_df


# ===== CELL 3 =====
# EDA reads published training scenarios only and is skipped in the hidden rerun.
# Failure here must never block the submission, so it is wrapped.
if not IS_RERUN:
    try:
        buildings, weather = analyze_data(DATA_ROOT, Path("/kaggle/working/eda") if Path("/kaggle/working").exists() else Path("eda"))
        from IPython.display import display, Image
        display(buildings)
        display(weather)
        display(Image(filename=str(Path("/kaggle/working/eda/eda.png"))))
        display(Image(filename=str(Path("/kaggle/working/eda/floor_plan.png"))))
    except Exception as exc:
        print("EDA skipped:", repr(exc))

# ===== CELL 5 =====
"""First baseline: average-zone control, sensor comfort bands, decoded weather.

Builds on the Apache-2.0 organizer PIDWaterResetPolicy (see vendor LICENSE).
Uses only metadata and live observations; no scenario identities or weather files.
"""
import numpy as np
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class BaselinePolicy(PIDWaterResetPolicy):
    USE_SENSOR_BAND = True
    # Worst-zone aggregation increased loss in both full-length pilot scenarios.
    USE_WORST_ZONE = False

    @property
    def name(self):
        return "transfer"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        stats = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self.outside_mean = float(stats.get("sample_mean", 0.0))
        self.outside_std = float(stats.get("sample_std", 1.0))

    def _recirculation(self, obs, ctrl, u, low, high):
        # The parent's economizer expects kelvin; the observation is standardized.
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        decoded = obs.copy()
        decoded[idx] = obs[idx] * self.outside_std + self.outside_mean
        return super()._recirculation(decoded, ctrl, u, low, high)

    def get_action(self, obs):
        action = super().get_action(obs)
        # Enforce the gateway's dtype, shape and bounds contract.
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action


# ===== CELL 6 =====
"""Main controller: the first baseline plus a cooling-run hold for air handlers.

The organizer AHU loop stops a unit as soon as its average-zone error returns
inside the comfort band, then restarts it at the next excursion. In logs each
AHU switched 50-107 times per day. Stopping a unit that was cooling turned out
to be the baseline's main loss in hot climates. This controller keeps a unit
that was started for cooling running at its last commands for at least
HOLD_STEPS steps. Heating-season runs are unchanged.

Same behaviour as `min_on_cool_<HOLD_STEPS>` in solution/research_policy.py.
Uses only metadata and live observations; no building or weather identities.
"""
import numpy as np


class CoolingHoldPolicy(BaselinePolicy):
    HOLD_STEPS = 288
    # Outside work hours use (eco low, comfort high): no overnight drift up to
    # the eco ceiling in hot climates, full heating setback kept in cold ones.
    # Idea from the public "Neurips 2026 Smart Buildings Smartpolicy" notebook.
    NIGHT_COOL_CEILING = True

    @property
    def name(self):
        return f"cooling_hold_{self.HOLD_STEPS}"

    def _base_action(self, obs):
        if not self.NIGHT_COOL_CEILING:
            return super().get_action(obs)
        hour, day = self._get_hour_and_day(obs)
        if day < 5 and self.morning_start_hour <= hour < self.evening_start_hour:
            return super().get_action(obs)
        saved = self.eco_temp_window
        self.eco_temp_window = (saved[0], self.comfort_temp_window[1])
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)

    def get_action(self, obs):
        action = self._base_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            run = ctrl["run_discrete_idx"]
            # Supply temperature first: its sign tells cooling from heating.
            idx = [j for j in (ctrl["temp_idx"], ctrl["pressure_idx"], ctrl.get("recirc_idx"))
                   if j is not None]
            if (not disc[run] and self._on_steps[i] > 0
                    and self._on_steps[i] < self.HOLD_STEPS and self._last_on[i][0] < 0):
                disc[run] = 1
                cont[idx] = self._last_on[i]
            if disc[run]:
                self._on_steps[i] += 1
                self._last_on[i] = cont[idx].copy()
            else:
                self._on_steps[i] = 0
        return action


class ZonePIPolicy(CoolingHoldPolicy):
    """v5 controller with per-zone PI control to the band centre in work hours.

    A soft proportional zone loop stops the 2-4 K oscillation but leaves zones
    sitting steadily outside the band (no integral action). Here each zone runs
    a PI loop on (temperature - band centre) from one hour before occupancy to
    the end of the work day. One signal u in [-1, 1] drives both actuators:
    the damper opens with |u| (air is needed to cool and to carry reheat) and
    the reheat valve opens with -u. AHUs are kept running in those hours so the
    zones have air to work with. Outside them, v5 logic is unchanged.
    """
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    KP = 0.2
    KI = 0.02
    PRESTART_H = 1
    FORCED_SAT = -0.7       # normalized supply temp for a forced-on AHU (~15 C)
    FORCED_PRESSURE = -0.6  # normalized static pressure for a forced-on AHU
    KEY = "zone_pi"

    @property
    def name(self):
        return self.KEY

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._zone_integral = {}
        self._zone_u = {}
        self._pi_active = False

    def _pi_hours(self, obs):
        hour, day = self._get_hour_and_day(obs)
        return day < 5 and self.morning_start_hour - self.PRESTART_H <= hour < self.evening_start_hour

    def _command_zones(self, continuous, zones, obs, low, high):
        if not self._pi_active:
            for vav_id, _, _ in zones:
                self._zone_integral[vav_id] = 0.0
            return super()._command_zones(continuous, zones, obs, low, high)
        lo_c, hi_c = self.comfort_temp_window
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, lo_c, hi_c)
            # Before occupancy the sensor band is still the eco band; aim at comfort.
            if hi - lo > (hi_c - lo_c) + 1e-6:
                lo, hi = lo_c, hi_c
            err = self._denormalize_zone_temp(obs[obs_idx]) - self._zone_target(obs, lo, hi)
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                integral = float(np.clip(integral + err, -1.0 / self.KI, 1.0 / self.KI))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            self._max_heat = max(self._max_heat, -u)
            self._zone_u[vav_id] = u
            gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * max(0.0, -u) - 1.0

    MIN_DAMPER = 0.1
    # Damper opening per unit of heating demand (reheated air costs gas).
    HEAT_DAMPER_GAIN = 1.0

    def _zone_target(self, obs, lo, hi):
        """Temperature each zone's PI loop aims at; the band centre by default."""
        return 0.5 * (lo + hi)

    def _base_action(self, obs):
        self._pi_active = self._pi_hours(obs)
        self._max_heat = 0.0
        self._forced = []
        self._zone_u = {}
        action = super()._base_action(obs)
        if self._pi_active:
            cont, disc = action["continuous_action"], action["discrete_action"]
            # Zone reheat needs hot water even when no AHU loop asks for heat.
            if self.water_temp_idx is not None and self._max_heat > 0:
                cont[self.water_temp_idx] = max(cont[self.water_temp_idx],
                                                np.float32(2.0 * self._max_heat - 1.0))
                if self.water_pressure_idx is not None:
                    cont[self.water_pressure_idx] = 1.0
            for i, ctrl in enumerate(self.ahu_controllers):
                if ctrl["zone_obs_indices"] and not disc[ctrl["run_discrete_idx"]]:
                    self._forced.append(i)
                    disc[ctrl["run_discrete_idx"]] = 1
                    cont[ctrl["temp_idx"]] = self.FORCED_SAT
                    if ctrl["pressure_idx"] is not None:
                        cont[ctrl["pressure_idx"]] = self.FORCED_PRESSURE
            if 0 <= self.water_discrete_idx < self.n_discrete:
                disc[self.water_discrete_idx] = 1
        return action

    def get_action(self, obs):
        action = super().get_action(obs)
        # A forced-on unit is not a cooling run; do not let the hold extend it.
        for i in self._forced:
            if self._last_on[i] is not None:
                self._last_on[i][0] = 0.0
        return action


# ===== CELL 7 =====
from kaggle_evaluation import smart_buildings_inference_server


class SubmissionPolicy(ZonePIPolicy, smart_buildings_inference_server.Policy):
    KP = 0.1
    KI = 0.005
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    FORCED_PRESSURE = -1.0

# ===== CELL 9 =====
smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=str(DATA_ROOT),
)