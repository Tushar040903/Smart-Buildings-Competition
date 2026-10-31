"""Generate holdout buildings and scenarios for local transfer validation.

Creates buildings with seeds 500+ (gitignored) and pairs them with weather
strips including minneapolis (shipped but unused in the official 30).

Usage:
    python scripts/generate_holdout.py --count 6
    python scripts/generate_holdout.py --count 6 --weathers fairbanks_1 phoenix_6 minneapolis_1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _shrink_buildings_toward_training_size(buildings_dir: Path, names: list[str]) -> None:
    """Cap width/height near shipped training buildings (see building_train_1)."""
    for name in names:
        path = buildings_dir / f"{name}.yaml"
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        fp = data.setdefault("floor_plan", {})
        w, h = int(fp.get("width", 400)), int(fp.get("height", 200))
        # Scale down if either edge exceeds the shipped envelope.
        scale = min(1.0, 400 / max(w, 1), 220 / max(h, 1))
        if scale < 1.0:
            fp["width"] = max(280, int(w * scale))
            fp["height"] = max(160, int(h * scale))
            fp["min_room_size"] = min(int(fp.get("min_room_size", 33)), 40)
            fp["max_room_size"] = min(int(fp.get("max_room_size", 48)), 55)
            data.setdefault("ahu", {})["max_zones_per_ahu"] = 10
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            print(f"  shrunk {name}: {w}x{h} -> {fp['width']}x{fp['height']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-seed", type=int, default=500)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument(
        "--weathers",
        nargs="+",
        default=["fairbanks_1", "phoenix_6", "singapore_1", "minneapolis_1", "denver_2"],
    )
    ap.add_argument("--skip-generate-buildings", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_generate_buildings:
        cmd = [
            py, "-m", "smart_buildings.run_floor_generator",
            "--start-seed", str(args.start_seed),
            "--count", str(args.count),
        ]
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=ROOT)

    buildings = [
        f"building_{args.start_seed + i:04d}" for i in range(args.count)
    ]
    # Shrink oversized footprints toward the shipped training range so
    # scenario generation / rollouts stay tractable (AHU count scales with
    # zone count; default generator footprints are much larger).
    _shrink_buildings_toward_training_size(ROOT / "smart_buildings" / "buildings", buildings)
    cmd = [
        py, "-m", "smart_buildings.run_scenario_generator",
        "--buildings", *buildings,
        "--weathers", *args.weathers,
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    print(f"Generated {len(buildings)} buildings x {len(args.weathers)} weathers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
