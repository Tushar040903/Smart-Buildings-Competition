# ===== CELL 1 =====
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

candidates = [Path('/kaggle/input/competitions/neurips-2026-smart-buildings'),
              Path('/kaggle/input/neurips-2026-smart-buildings')]
if os.getenv('SMART_BUILDINGS_DATA_ROOT'):
    candidates.insert(0, Path(os.environ['SMART_BUILDINGS_DATA_ROOT']))
DATA_ROOT = next((p for p in candidates if (p / 'smart_buildings').exists()), None)
if DATA_ROOT is None:
    raise FileNotFoundError('Attach competition data or set SMART_BUILDINGS_DATA_ROOT')
sys.path.insert(0, str(DATA_ROOT))
sys.path.insert(0, str(DATA_ROOT / 'kaggle_evaluation'))
IS_RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
print('Competition data located; hidden rerun:', IS_RERUN)


# ===== CELL 3 =====
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


# ===== CELL 4 =====
if not IS_RERUN:
    buildings, weather = analyze_data(DATA_ROOT, Path.cwd() / 'eda')
    from IPython.display import display, Image
    display(buildings)
    display(weather)
    display(Image(filename='eda/eda.png'))
    display(Image(filename='eda/floor_plan.png'))


# ===== CELL 6 =====
pilot = pd.DataFrame(json.loads('[{"policy": "min_on_cool_288", "building_train_1/atlanta_2": 0.021844, "building_train_1/denver_7": 0.024986, "building_train_1/fairbanks_1": 0.214082, "building_train_1/phoenix_5": 0.059433, "building_train_1/singapore_8": 0.024687, "building_train_2/atlanta_3": 0.016204, "building_train_2/denver_8": 0.034157, "building_train_2/fairbanks_2": 0.070825, "building_train_2/phoenix_6": 0.034516, "building_train_2/singapore_1": 0.01804, "building_train_4/atlanta_5": 0.024324, "building_train_4/denver_2": 0.052013, "building_train_4/fairbanks_4": 0.024768, "building_train_4/phoenix_8": 0.022073, "building_train_4/singapore_3": 0.025551, "building_train_5/atlanta_6": 0.014049, "building_train_5/denver_3": 0.037197, "building_train_5/fairbanks_5": 0.012584, "building_train_5/phoenix_1": 0.026367, "building_train_5/singapore_4": 0.024586, "building_train_7/atlanta_8": 0.046477, "building_train_7/denver_5": 0.018875, "building_train_7/fairbanks_7": 0.053886, "building_train_7/phoenix_3": 0.02568, "building_train_7/singapore_6": 0.0215, "building_train_8/atlanta_1": 0.0318, "building_train_8/denver_6": 0.01564, "building_train_8/fairbanks_8": 0.077077, "building_train_8/phoenix_4": 0.044971, "building_train_8/singapore_7": 0.0191, "mean_improvement_pct": 45.280909, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.0 / -27.4 / -24.2"}, {"policy": "min_on_cool_96", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077077, "building_train_8/phoenix_4": 0.055115, "building_train_8/singapore_7": 0.019998, "mean_improvement_pct": 42.639779, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.0 / -25.1 / -23.5"}, {"policy": "min_on_cool_48", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077077, "building_train_8/phoenix_4": 0.058816, "building_train_8/singapore_7": 0.020033, "mean_improvement_pct": 41.863527, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.0 / -24.2 / -23.5"}, {"policy": "min_on_cool_24", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077077, "building_train_8/phoenix_4": 0.078474, "building_train_8/singapore_7": 0.02173, "mean_improvement_pct": 36.773736, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.0 / -19.7 / -22.3"}, {"policy": "min_on_cool_12", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077886, "building_train_8/phoenix_4": 0.098678, "building_train_8/singapore_7": 0.021685, "mean_improvement_pct": 32.338253, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 2.2 / -15.0 / -22.3"}, {"policy": "min_on_cool_6", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077316, "building_train_8/phoenix_4": 0.102337, "building_train_8/singapore_7": 0.023848, "mean_improvement_pct": 30.4552, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.7 / -14.2 / -20.7"}, {"policy": "min_on_cool_3", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077297, "building_train_8/phoenix_4": 0.128013, "building_train_8/singapore_7": 0.030711, "mean_improvement_pct": 20.842412, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.6 / -8.2 / -15.7"}, {"policy": "min_on_6", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.094148, "building_train_8/phoenix_4": 0.105296, "building_train_8/singapore_7": 0.03669, "mean_improvement_pct": 14.355608, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 45.2 / -13.5 / -11.3"}, {"policy": "min_on_3", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.087441, "building_train_8/phoenix_4": 0.123693, "building_train_8/singapore_7": 0.035456, "mean_improvement_pct": 14.298793, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 27.5 / -9.2 / -12.2"}, {"policy": "null_2", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077393, "building_train_8/phoenix_4": 0.159066, "building_train_8/singapore_7": 0.052083, "mean_improvement_pct": 0.801494, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.9 / -1.1 / -0.0"}, {"policy": "null_1", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.076552, "building_train_8/phoenix_4": 0.162694, "building_train_8/singapore_7": 0.052084, "mean_improvement_pct": 0.425973, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / -1.4 / -0.2 / -0.0"}, {"policy": "null_3", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077011, "building_train_8/phoenix_4": 0.169492, "building_train_8/singapore_7": 0.05044, "mean_improvement_pct": -0.104937, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / -0.1 / 1.3 / -1.2"}, {"policy": "transfer", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077297, "building_train_8/phoenix_4": 0.163381, "building_train_8/singapore_7": 0.053782, "mean_improvement_pct": -1.122531, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 0.6 / -0.1 / 1.2"}, {"policy": "pressure_reset", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077847, "building_train_8/phoenix_4": 0.166517, "building_train_8/singapore_7": 0.052698, "mean_improvement_pct": -1.305578, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 2.1 / 0.7 / 0.4"}, {"policy": "min_on_2", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.077797, "building_train_8/phoenix_4": 0.157236, "building_train_8/singapore_7": 0.056332, "mean_improvement_pct": -1.718751, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 1.9 / -1.5 / 3.1"}, {"policy": "start_02", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.082354, "building_train_8/phoenix_4": 0.147884, "building_train_8/singapore_7": 0.063, "mean_improvement_pct": -6.051391, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 14.0 / -3.6 / 8.0"}, {"policy": "start_05", "building_train_1/atlanta_2": NaN, "building_train_1/denver_7": NaN, "building_train_1/fairbanks_1": NaN, "building_train_1/phoenix_5": NaN, "building_train_1/singapore_8": NaN, "building_train_2/atlanta_3": NaN, "building_train_2/denver_8": NaN, "building_train_2/fairbanks_2": NaN, "building_train_2/phoenix_6": NaN, "building_train_2/singapore_1": NaN, "building_train_4/atlanta_5": NaN, "building_train_4/denver_2": NaN, "building_train_4/fairbanks_4": NaN, "building_train_4/phoenix_8": NaN, "building_train_4/singapore_3": NaN, "building_train_5/atlanta_6": NaN, "building_train_5/denver_3": NaN, "building_train_5/fairbanks_5": NaN, "building_train_5/phoenix_1": NaN, "building_train_5/singapore_4": NaN, "building_train_7/atlanta_8": NaN, "building_train_7/denver_5": NaN, "building_train_7/fairbanks_7": NaN, "building_train_7/phoenix_3": NaN, "building_train_7/singapore_6": NaN, "building_train_8/atlanta_1": NaN, "building_train_8/denver_6": NaN, "building_train_8/fairbanks_8": 0.089897, "building_train_8/phoenix_4": 0.184582, "building_train_8/singapore_7": 0.112904, "mean_improvement_pct": -48.718099, "z_by_scenario": "nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / nan / 34.0 / 4.8 / 44.6"}]'))
if not IS_RERUN:
    display(pilot)


# ===== CELL 8 =====
full = pd.DataFrame(json.loads('[{"scenario": "building_train_1/atlanta_2", "policy": "min_on_cool_6", "steps": 8640, "score": 0.023757, "comfort_loss": 0.019626, "energy_loss": 0.002198, "carbon_loss": 0.001933}, {"scenario": "building_train_1/atlanta_2", "policy": "transfer", "steps": 8640, "score": 0.024181, "comfort_loss": 0.019969, "energy_loss": 0.002269, "carbon_loss": 0.001942}, {"scenario": "building_train_1/denver_7", "policy": "min_on_cool_6", "steps": 8640, "score": 0.029103, "comfort_loss": 0.028222, "energy_loss": 0.000431, "carbon_loss": 0.00045}, {"scenario": "building_train_1/denver_7", "policy": "transfer", "steps": 8640, "score": 0.029976, "comfort_loss": 0.029086, "energy_loss": 0.000445, "carbon_loss": 0.000445}, {"scenario": "building_train_1/fairbanks_1", "policy": "min_on_cool_6", "steps": 8640, "score": 0.214244, "comfort_loss": 0.070473, "energy_loss": 0.055038, "carbon_loss": 0.088733}, {"scenario": "building_train_1/fairbanks_1", "policy": "transfer", "steps": 8640, "score": 0.212925, "comfort_loss": 0.069706, "energy_loss": 0.054741, "carbon_loss": 0.088478}, {"scenario": "building_train_1/phoenix_5", "policy": "min_on_cool_6", "steps": 8640, "score": 0.104265, "comfort_loss": 0.052861, "energy_loss": 0.037869, "carbon_loss": 0.013536}, {"scenario": "building_train_1/phoenix_5", "policy": "transfer", "steps": 8640, "score": 0.136072, "comfort_loss": 0.074278, "energy_loss": 0.045083, "carbon_loss": 0.016711}, {"scenario": "building_train_1/singapore_8", "policy": "min_on_cool_6", "steps": 8640, "score": 0.025284, "comfort_loss": 0.015158, "energy_loss": 0.008101, "carbon_loss": 0.002026}, {"scenario": "building_train_1/singapore_8", "policy": "transfer", "steps": 8640, "score": 0.028131, "comfort_loss": 0.017798, "energy_loss": 0.008262, "carbon_loss": 0.002071}, {"scenario": "building_train_2/atlanta_3", "policy": "min_on_cool_6", "steps": 8640, "score": 0.019147, "comfort_loss": 0.014722, "energy_loss": 0.003566, "carbon_loss": 0.000859}, {"scenario": "building_train_2/atlanta_3", "policy": "transfer", "steps": 8640, "score": 0.020152, "comfort_loss": 0.015595, "energy_loss": 0.003674, "carbon_loss": 0.000884}, {"scenario": "building_train_2/denver_8", "policy": "min_on_cool_6", "steps": 8640, "score": 0.038612, "comfort_loss": 0.032884, "energy_loss": 0.002491, "carbon_loss": 0.003237}, {"scenario": "building_train_2/denver_8", "policy": "transfer", "steps": 8640, "score": 0.039837, "comfort_loss": 0.034068, "energy_loss": 0.002537, "carbon_loss": 0.003232}, {"scenario": "building_train_2/fairbanks_2", "policy": "min_on_cool_6", "steps": 8640, "score": 0.077485, "comfort_loss": 0.058128, "energy_loss": 0.007402, "carbon_loss": 0.011955}, {"scenario": "building_train_2/fairbanks_2", "policy": "transfer", "steps": 8640, "score": 0.079776, "comfort_loss": 0.060406, "energy_loss": 0.007428, "carbon_loss": 0.011942}, {"scenario": "building_train_2/phoenix_6", "policy": "min_on_cool_6", "steps": 8640, "score": 0.053152, "comfort_loss": 0.029955, "energy_loss": 0.017501, "carbon_loss": 0.005696}, {"scenario": "building_train_2/phoenix_6", "policy": "transfer", "steps": 8640, "score": 0.070571, "comfort_loss": 0.040747, "energy_loss": 0.022018, "carbon_loss": 0.007807}, {"scenario": "building_train_2/singapore_1", "policy": "min_on_cool_6", "steps": 8640, "score": 0.021724, "comfort_loss": 0.01415, "energy_loss": 0.00608, "carbon_loss": 0.001494}, {"scenario": "building_train_2/singapore_1", "policy": "transfer", "steps": 8640, "score": 0.026442, "comfort_loss": 0.017387, "energy_loss": 0.007053, "carbon_loss": 0.002002}, {"scenario": "building_train_4/atlanta_5", "policy": "min_on_cool_6", "steps": 8640, "score": 0.026179, "comfort_loss": 0.016057, "energy_loss": 0.008194, "carbon_loss": 0.001928}, {"scenario": "building_train_4/atlanta_5", "policy": "transfer", "steps": 8640, "score": 0.028679, "comfort_loss": 0.018379, "energy_loss": 0.008352, "carbon_loss": 0.001948}, {"scenario": "building_train_4/denver_2", "policy": "min_on_cool_6", "steps": 8640, "score": 0.054021, "comfort_loss": 0.04858, "energy_loss": 0.002072, "carbon_loss": 0.00337}, {"scenario": "building_train_4/denver_2", "policy": "transfer", "steps": 8640, "score": 0.054502, "comfort_loss": 0.049156, "energy_loss": 0.002036, "carbon_loss": 0.00331}, {"scenario": "building_train_4/fairbanks_4", "policy": "min_on_cool_6", "steps": 8640, "score": 0.030803, "comfort_loss": 0.029361, "energy_loss": 0.000969, "carbon_loss": 0.000473}, {"scenario": "building_train_4/fairbanks_4", "policy": "transfer", "steps": 8640, "score": 0.031825, "comfort_loss": 0.030207, "energy_loss": 0.00111, "carbon_loss": 0.000508}, {"scenario": "building_train_4/phoenix_8", "policy": "min_on_cool_6", "steps": 8640, "score": 0.028171, "comfort_loss": 0.026614, "energy_loss": 0.001167, "carbon_loss": 0.00039}, {"scenario": "building_train_4/phoenix_8", "policy": "transfer", "steps": 8640, "score": 0.029013, "comfort_loss": 0.027287, "energy_loss": 0.001302, "carbon_loss": 0.000424}, {"scenario": "building_train_4/singapore_3", "policy": "min_on_cool_6", "steps": 8640, "score": 0.026857, "comfort_loss": 0.015657, "energy_loss": 0.009063, "carbon_loss": 0.002137}, {"scenario": "building_train_4/singapore_3", "policy": "transfer", "steps": 8640, "score": 0.035134, "comfort_loss": 0.021013, "energy_loss": 0.0109, "carbon_loss": 0.003221}, {"scenario": "building_train_5/atlanta_6", "policy": "min_on_cool_6", "steps": 8640, "score": 0.021838, "comfort_loss": 0.014134, "energy_loss": 0.00608, "carbon_loss": 0.001624}, {"scenario": "building_train_5/atlanta_6", "policy": "transfer", "steps": 8640, "score": 0.03416, "comfort_loss": 0.022188, "energy_loss": 0.008863, "carbon_loss": 0.003109}, {"scenario": "building_train_5/denver_3", "policy": "min_on_cool_6", "steps": 8640, "score": 0.04218, "comfort_loss": 0.037471, "energy_loss": 0.001789, "carbon_loss": 0.00292}, {"scenario": "building_train_5/denver_3", "policy": "transfer", "steps": 8640, "score": 0.043519, "comfort_loss": 0.038822, "energy_loss": 0.001793, "carbon_loss": 0.002905}, {"scenario": "building_train_5/fairbanks_5", "policy": "min_on_cool_6", "steps": 8640, "score": 0.01637, "comfort_loss": 0.014359, "energy_loss": 0.001613, "carbon_loss": 0.000399}, {"scenario": "building_train_5/fairbanks_5", "policy": "transfer", "steps": 8640, "score": 0.016909, "comfort_loss": 0.01477, "energy_loss": 0.001716, "carbon_loss": 0.000423}, {"scenario": "building_train_5/phoenix_1", "policy": "min_on_cool_6", "steps": 8640, "score": 0.032137, "comfort_loss": 0.029887, "energy_loss": 0.001003, "carbon_loss": 0.001247}, {"scenario": "building_train_5/phoenix_1", "policy": "transfer", "steps": 8640, "score": 0.033239, "comfort_loss": 0.030978, "energy_loss": 0.001016, "carbon_loss": 0.001246}, {"scenario": "building_train_5/singapore_4", "policy": "min_on_cool_6", "steps": 8640, "score": 0.078257, "comfort_loss": 0.044401, "energy_loss": 0.023017, "carbon_loss": 0.010839}, {"scenario": "building_train_5/singapore_4", "policy": "transfer", "steps": 8640, "score": 0.107665, "comfort_loss": 0.061349, "energy_loss": 0.030598, "carbon_loss": 0.015718}, {"scenario": "building_train_7/atlanta_8", "policy": "min_on_cool_6", "steps": 8640, "score": 0.048211, "comfort_loss": 0.039145, "energy_loss": 0.003381, "carbon_loss": 0.005684}, {"scenario": "building_train_7/atlanta_8", "policy": "transfer", "steps": 8640, "score": 0.048465, "comfort_loss": 0.039426, "energy_loss": 0.003368, "carbon_loss": 0.005671}, {"scenario": "building_train_7/denver_5", "policy": "min_on_cool_6", "steps": 8640, "score": 0.029159, "comfort_loss": 0.018946, "energy_loss": 0.007654, "carbon_loss": 0.00256}, {"scenario": "building_train_7/denver_5", "policy": "transfer", "steps": 8640, "score": 0.036026, "comfort_loss": 0.023814, "energy_loss": 0.009045, "carbon_loss": 0.003166}, {"scenario": "building_train_7/fairbanks_7", "policy": "min_on_cool_6", "steps": 8640, "score": 0.053854, "comfort_loss": 0.042241, "energy_loss": 0.004153, "carbon_loss": 0.00746}, {"scenario": "building_train_7/fairbanks_7", "policy": "transfer", "steps": 8640, "score": 0.053778, "comfort_loss": 0.042199, "energy_loss": 0.004147, "carbon_loss": 0.007433}, {"scenario": "building_train_7/phoenix_3", "policy": "min_on_cool_6", "steps": 8640, "score": 0.036567, "comfort_loss": 0.022531, "energy_loss": 0.010612, "carbon_loss": 0.003424}, {"scenario": "building_train_7/phoenix_3", "policy": "transfer", "steps": 8640, "score": 0.044925, "comfort_loss": 0.028412, "energy_loss": 0.012163, "carbon_loss": 0.00435}, {"scenario": "building_train_7/singapore_6", "policy": "min_on_cool_6", "steps": 8640, "score": 0.024553, "comfort_loss": 0.014502, "energy_loss": 0.008049, "carbon_loss": 0.002002}, {"scenario": "building_train_7/singapore_6", "policy": "transfer", "steps": 8640, "score": 0.034823, "comfort_loss": 0.021331, "energy_loss": 0.010233, "carbon_loss": 0.003259}, {"scenario": "building_train_8/atlanta_1", "policy": "min_on_cool_6", "steps": 8640, "score": 0.032359, "comfort_loss": 0.024291, "energy_loss": 0.003056, "carbon_loss": 0.005012}, {"scenario": "building_train_8/atlanta_1", "policy": "transfer", "steps": 8640, "score": 0.032533, "comfort_loss": 0.024467, "energy_loss": 0.003069, "carbon_loss": 0.004997}, {"scenario": "building_train_8/denver_6", "policy": "min_on_cool_6", "steps": 8640, "score": 0.019112, "comfort_loss": 0.014077, "energy_loss": 0.004056, "carbon_loss": 0.000978}, {"scenario": "building_train_8/denver_6", "policy": "transfer", "steps": 8640, "score": 0.020257, "comfort_loss": 0.015097, "energy_loss": 0.004163, "carbon_loss": 0.000997}, {"scenario": "building_train_8/fairbanks_8", "policy": "min_on_cool_6", "steps": 8640, "score": 0.077316, "comfort_loss": 0.031317, "energy_loss": 0.016066, "carbon_loss": 0.029934}, {"scenario": "building_train_8/fairbanks_8", "policy": "transfer", "steps": 8640, "score": 0.077297, "comfort_loss": 0.031107, "energy_loss": 0.016144, "carbon_loss": 0.030046}, {"scenario": "building_train_8/phoenix_4", "policy": "min_on_cool_6", "steps": 8640, "score": 0.102337, "comfort_loss": 0.05544, "energy_loss": 0.03332, "carbon_loss": 0.013577}, {"scenario": "building_train_8/phoenix_4", "policy": "transfer", "steps": 8640, "score": 0.163381, "comfort_loss": 0.09317, "energy_loss": 0.048682, "carbon_loss": 0.021529}, {"scenario": "building_train_8/singapore_7", "policy": "min_on_cool_6", "steps": 8640, "score": 0.023848, "comfort_loss": 0.01431, "energy_loss": 0.007552, "carbon_loss": 0.001986}, {"scenario": "building_train_8/singapore_7", "policy": "transfer", "steps": 8640, "score": 0.053782, "comfort_loss": 0.032004, "energy_loss": 0.015029, "carbon_loss": 0.006748}, {"scenario": "building_train_1/atlanta_2", "policy": "min_on_cool_288", "steps": 8640, "score": 0.021844, "comfort_loss": 0.01768, "energy_loss": 0.002205, "carbon_loss": 0.001959}, {"scenario": "building_train_1/denver_7", "policy": "min_on_cool_288", "steps": 8640, "score": 0.024986, "comfort_loss": 0.023832, "energy_loss": 0.000537, "carbon_loss": 0.000616}, {"scenario": "building_train_1/fairbanks_1", "policy": "min_on_cool_288", "steps": 8640, "score": 0.214082, "comfort_loss": 0.070397, "energy_loss": 0.055012, "carbon_loss": 0.088673}, {"scenario": "building_train_1/phoenix_5", "policy": "min_on_cool_288", "steps": 8640, "score": 0.059433, "comfort_loss": 0.025426, "energy_loss": 0.026264, "carbon_loss": 0.007743}, {"scenario": "building_train_1/singapore_8", "policy": "min_on_cool_288", "steps": 8640, "score": 0.024687, "comfort_loss": 0.014527, "energy_loss": 0.008139, "carbon_loss": 0.002021}, {"scenario": "building_train_2/atlanta_3", "policy": "min_on_cool_288", "steps": 8640, "score": 0.016204, "comfort_loss": 0.011179, "energy_loss": 0.00399, "carbon_loss": 0.001036}, {"scenario": "building_train_2/denver_8", "policy": "min_on_cool_288", "steps": 8640, "score": 0.034157, "comfort_loss": 0.027327, "energy_loss": 0.002823, "carbon_loss": 0.004006}, {"scenario": "building_train_2/fairbanks_2", "policy": "min_on_cool_288", "steps": 8640, "score": 0.070825, "comfort_loss": 0.048753, "energy_loss": 0.008457, "carbon_loss": 0.013616}, {"scenario": "building_train_2/phoenix_6", "policy": "min_on_cool_288", "steps": 8640, "score": 0.034516, "comfort_loss": 0.016715, "energy_loss": 0.013898, "carbon_loss": 0.003904}, {"scenario": "building_train_2/singapore_1", "policy": "min_on_cool_288", "steps": 8640, "score": 0.01804, "comfort_loss": 0.009663, "energy_loss": 0.006722, "carbon_loss": 0.001655}, {"scenario": "building_train_4/atlanta_5", "policy": "min_on_cool_288", "steps": 8640, "score": 0.024324, "comfort_loss": 0.014176, "energy_loss": 0.008235, "carbon_loss": 0.001913}, {"scenario": "building_train_4/denver_2", "policy": "min_on_cool_288", "steps": 8640, "score": 0.052013, "comfort_loss": 0.046023, "energy_loss": 0.00225, "carbon_loss": 0.003739}, {"scenario": "building_train_4/fairbanks_4", "policy": "min_on_cool_288", "steps": 8640, "score": 0.024768, "comfort_loss": 0.023271, "energy_loss": 0.000926, "carbon_loss": 0.000571}, {"scenario": "building_train_4/phoenix_8", "policy": "min_on_cool_288", "steps": 8640, "score": 0.022073, "comfort_loss": 0.020401, "energy_loss": 0.001196, "carbon_loss": 0.000476}, {"scenario": "building_train_4/singapore_3", "policy": "min_on_cool_288", "steps": 8640, "score": 0.025551, "comfort_loss": 0.014363, "energy_loss": 0.009088, "carbon_loss": 0.002101}, {"scenario": "building_train_5/atlanta_6", "policy": "min_on_cool_288", "steps": 8640, "score": 0.014049, "comfort_loss": 0.006861, "energy_loss": 0.005826, "carbon_loss": 0.001363}, {"scenario": "building_train_5/denver_3", "policy": "min_on_cool_288", "steps": 8640, "score": 0.037197, "comfort_loss": 0.03204, "energy_loss": 0.001973, "carbon_loss": 0.003183}, {"scenario": "building_train_5/fairbanks_5", "policy": "min_on_cool_288", "steps": 8640, "score": 0.012584, "comfort_loss": 0.010512, "energy_loss": 0.001652, "carbon_loss": 0.000421}, {"scenario": "building_train_5/phoenix_1", "policy": "min_on_cool_288", "steps": 8640, "score": 0.026367, "comfort_loss": 0.02369, "energy_loss": 0.001181, "carbon_loss": 0.001495}, {"scenario": "building_train_5/singapore_4", "policy": "min_on_cool_288", "steps": 8640, "score": 0.024586, "comfort_loss": 0.011622, "energy_loss": 0.010014, "carbon_loss": 0.00295}, {"scenario": "building_train_7/atlanta_8", "policy": "min_on_cool_288", "steps": 8640, "score": 0.046477, "comfort_loss": 0.037122, "energy_loss": 0.003493, "carbon_loss": 0.005863}, {"scenario": "building_train_7/denver_5", "policy": "min_on_cool_288", "steps": 8640, "score": 0.018875, "comfort_loss": 0.010676, "energy_loss": 0.006456, "carbon_loss": 0.001743}, {"scenario": "building_train_7/fairbanks_7", "policy": "min_on_cool_288", "steps": 8640, "score": 0.053886, "comfort_loss": 0.041918, "energy_loss": 0.004285, "carbon_loss": 0.007683}, {"scenario": "building_train_7/phoenix_3", "policy": "min_on_cool_288", "steps": 8640, "score": 0.02568, "comfort_loss": 0.013867, "energy_loss": 0.009322, "carbon_loss": 0.002492}, {"scenario": "building_train_7/singapore_6", "policy": "min_on_cool_288", "steps": 8640, "score": 0.0215, "comfort_loss": 0.010992, "energy_loss": 0.008487, "carbon_loss": 0.002021}, {"scenario": "building_train_8/atlanta_1", "policy": "min_on_cool_288", "steps": 8640, "score": 0.0318, "comfort_loss": 0.023459, "energy_loss": 0.003166, "carbon_loss": 0.005175}, {"scenario": "building_train_8/denver_6", "policy": "min_on_cool_288", "steps": 8640, "score": 0.01564, "comfort_loss": 0.009264, "energy_loss": 0.004996, "carbon_loss": 0.001379}, {"scenario": "building_train_8/fairbanks_8", "policy": "min_on_cool_288", "steps": 8640, "score": 0.077077, "comfort_loss": 0.031436, "energy_loss": 0.015926, "carbon_loss": 0.029716}, {"scenario": "building_train_8/phoenix_4", "policy": "min_on_cool_288", "steps": 8640, "score": 0.044971, "comfort_loss": 0.018981, "energy_loss": 0.019934, "carbon_loss": 0.006056}, {"scenario": "building_train_8/singapore_7", "policy": "min_on_cool_288", "steps": 8640, "score": 0.0191, "comfort_loss": 0.00903, "energy_loss": 0.008106, "carbon_loss": 0.001963}]'))
CANDIDATE = 'min_on_cool_288'
if not IS_RERUN and len(full):
    scores = full.pivot(index='scenario', columns='policy', values='score')
    cols = [c for c in ('transfer', CANDIDATE) if c in scores]
    scores = scores[cols]
    scores['pilot_building'] = scores.index.str.startswith('building_train_8/')
    if len(cols) == 2:
        scores['change_pct'] = 100 * (scores[CANDIDATE] - scores['transfer']) / scores['transfer']
    display(scores.round(4))
    climate = scores[cols].groupby(scores.index.str.split('/').str[1].str.replace(r'_\d+$', '', regex=True)).mean()
    display(climate.round(4))
    summary = pd.DataFrame({
        'all 30': scores[cols].mean(),
        'non-pilot 25': scores.loc[~scores.pilot_building, cols].mean()})
    display(summary.round(5))
    ax = climate.plot.bar(figsize=(9, 4), rot=0)
    ax.set(title='Mean episode score by climate (lower is better)', ylabel='Score', xlabel='Climate')
    plt.tight_layout(); plt.show()


# ===== CELL 10 =====
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


# ===== CELL 11 =====
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
            err = self._denormalize_zone_temp(obs[obs_idx]) - 0.5 * (lo + hi)
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                integral = float(np.clip(integral + err, -1.0 / self.KI, 1.0 / self.KI))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            self._max_heat = max(self._max_heat, -u)
            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * max(0.0, -u) - 1.0

    MIN_DAMPER = 0.1

    def _base_action(self, obs):
        self._pi_active = self._pi_hours(obs)
        self._max_heat = 0.0
        self._forced = []
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


# ===== CELL 12 =====
from kaggle_evaluation import smart_buildings_inference_server

class SubmissionPolicy(ZonePIPolicy, smart_buildings_inference_server.Policy):
    KP = 0.1
    KI = 0.005
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True


# ===== CELL 14 =====
smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=str(DATA_ROOT),
)
