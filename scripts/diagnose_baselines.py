"""Diagnose shipped baseline returns by climate and building.

Reads baseline_returns.json (no simulator run). Prints where pid_water loses
to pid, and climate aggregates useful for Layer A design.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baseline_returns.json"
STEPS = 8640
POLICIES = ("bang_bang", "fixed_setpoint", "pid", "pid_water")


def station(weather: str) -> str:
    return weather.rsplit("_", 1)[0]


def main() -> None:
    data = json.loads(BASELINES.read_text())
    by_station: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_building: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pid_beats_water: list[tuple[str, float, float]] = []

    for key, returns in sorted(data.items()):
        building, weather = key.split("/")
        st = station(weather)
        for p in POLICIES:
            r = returns[p]
            by_station[st][p].append(r)
            by_building[building][p].append(r)
        if returns["pid"] > returns["pid_water"]:  # less negative is better
            pid_beats_water.append(
                (key, returns["pid"], returns["pid_water"])
            )

    print("=== Leaderboard metric (MAE of mean per-step reward) ===")
    for p in POLICIES:
        vals = [abs(data[k][p]) / STEPS for k in data]
        print(f"  {p:<14} {statistics.mean(vals):.5f}")

    print("\n=== By climate station (mean total reward; higher/less-neg better) ===")
    header = f"{'station':<12}" + "".join(f"{p:>14}" for p in POLICIES)
    print(header)
    for st in sorted(by_station):
        row = f"{st:<12}"
        for p in POLICIES:
            row += f"{statistics.mean(by_station[st][p]):>14.1f}"
        print(row)

    print("\n=== Scenarios where pid beats pid_water (cold-climate water reset risk) ===")
    if not pid_beats_water:
        print("  (none)")
    for key, pid_r, water_r in sorted(pid_beats_water, key=lambda x: x[2] - x[1]):
        print(f"  {key:<36} pid={pid_r:>10.1f}  pid_water={water_r:>10.1f}  "
              f"gap={water_r - pid_r:>8.1f}")

    print("\n=== Worst pid_water scenarios (most negative return) ===")
    worst = sorted(data.items(), key=lambda kv: kv[1]["pid_water"])[:8]
    for key, r in worst:
        print(f"  {key:<36} pid_water={r['pid_water']:>10.1f}  "
              f"pid={r['pid']:>10.1f}  bang={r['bang_bang']:>10.1f}")

    print("\n=== Notes for Layer A ===")
    print("  - Fairbanks: SAT cooling inert ~31% of hours; favor reheat + water.")
    print("  - Phoenix: heat spikes; economizer + pressure matter.")
    print("  - pid_water water-reset can hurt when heat is unreachable (outliers).")
    print("  - Enable USE_WORST_ZONE + USE_SENSOR_BAND (shipped pid_water has both False).")


if __name__ == "__main__":
    main()
