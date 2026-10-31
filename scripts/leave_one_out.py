"""Leave-one-building-out / leave-one-climate-out scoring helpers.

Does not re-simulate. Given a results tree (from evaluate.py) and optional
baseline_returns.json, prints LOBO / LOCO leaderboard scores so you can tune
without fitting the full public 30.

Usage:
    python scripts/leave_one_out.py smart_buildings/results/my_run --policy transfer
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

STEPS = 8640


def load_tree(root: Path, policy: str) -> dict[str, float]:
    out = {}
    for f in root.glob("building_*/*/" + policy + "/summary.json"):
        key = f"{f.parts[-4]}/{f.parts[-3]}"
        out[key] = float(json.loads(f.read_text())["total_reward"])
    return out


def score(returns: dict[str, float]) -> float:
    if not returns:
        return float("nan")
    return statistics.mean(abs(r) / STEPS for r in returns.values())


def station(weather: str) -> str:
    return weather.rsplit("_", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--policy", default="transfer")
    ap.add_argument(
        "--baselines",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "baseline_returns.json",
    )
    args = ap.parse_args()

    tree = load_tree(args.results_dir, args.policy)
    if not tree:
        print(f"no results for {args.policy} under {args.results_dir}")
        return 1

    print(f"=== {args.policy}: full score {score(tree):.5f} over {len(tree)} scenarios ===")

    by_building: dict[str, dict[str, float]] = defaultdict(dict)
    by_climate: dict[str, dict[str, float]] = defaultdict(dict)
    for key, r in tree.items():
        b, w = key.split("/")
        by_building[b][key] = r
        by_climate[station(w)][key] = r

    print("\n--- Leave-one-building-out (score on held-out building) ---")
    for b in sorted(by_building):
        held = by_building[b]
        rest = {k: v for k, v in tree.items() if not k.startswith(b + "/")}
        print(f"  hold out {b:<20} held={score(held):.5f} (n={len(held)})  "
              f"train={score(rest):.5f} (n={len(rest)})")

    print("\n--- Leave-one-climate-out (score on held-out climate) ---")
    for c in sorted(by_climate):
        held = by_climate[c]
        rest = {k: v for k, v in tree.items() if station(k.split("/")[1]) != c}
        print(f"  hold out {c:<12} held={score(held):.5f} (n={len(held)})  "
              f"train={score(rest):.5f} (n={len(rest)})")

    if args.baselines.exists():
        base = json.loads(args.baselines.read_text())
        print("\n--- vs pid_water on overlapping scenarios ---")
        overlap = [k for k in tree if k in base]
        if overlap:
            ours = statistics.mean(abs(tree[k]) / STEPS for k in overlap)
            water = statistics.mean(abs(base[k]["pid_water"]) / STEPS for k in overlap)
            print(f"  {args.policy}={ours:.5f}  pid_water={water:.5f}  "
                  f"delta={ours - water:+.5f} (negative = we win)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
