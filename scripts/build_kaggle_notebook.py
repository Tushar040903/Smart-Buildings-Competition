"""Build a self-contained Kaggle submission .py (paste into notebook cells).

Embeds pid.py + pid_plus.py + TransferPolicy so the notebook does not depend
on the starter-kit package layout beyond kaggle_evaluation.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "kaggle_submission.py"

PID = (ROOT / "smart_buildings" / "baselines" / "pid.py").read_text(encoding="utf-8")
PID_PLUS = (ROOT / "smart_buildings" / "baselines" / "pid_plus.py").read_text(
    encoding="utf-8"
)
TRANSFER = (ROOT / "smart_buildings" / "baselines" / "transfer_policy.py").read_text(
    encoding="utf-8"
)

# Rewrite imports so the embedded modules resolve against each other.
PID_PLUS_EMBED = PID_PLUS.replace(
    "from smart_buildings.baselines.pid import PIDPolicy",
    "# PIDPolicy defined above in this notebook cell\n",
)
TRANSFER_EMBED = TRANSFER.replace(
    "from smart_buildings.baselines.pid_plus import (\n"
    "    SUPPLY_AIR_MIN_K,\n"
    "    PIDWaterResetPolicy,\n"
    ")",
    "# SUPPLY_AIR_MIN_K, PIDWaterResetPolicy defined above\n",
).replace(
    "from smart_buildings.baselines.pid_plus import (\n"
    "    SUPPLY_AIR_MIN_K,\n"
    "    PIDWaterResetPolicy,\n"
    ")",
    "",
)

import re

def clean_embed(text: str) -> str:
    # Strip future imports (must only be at top of file)
    text = re.sub(r"from __future__ import annotations\s*\n", "", text)
    # Strip multiline imports first
    text = re.sub(r"from smart_buildings\.baselines\.[a-z_]+ import \([^\)]*\)\s*\n", "", text, flags=re.DOTALL)
    # Strip single-line package imports
    text = re.sub(r"from smart_buildings\.baselines\.[a-z_]+ import [^\n]+\n", "", text)
    return text

PID_EMBED = clean_embed(PID)
PID_PLUS_EMBED = clean_embed(PID_PLUS)
TRANSFER_EMBED = clean_embed(TRANSFER)

HEADER = '''\
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
'''

FOOTER = '''
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
'''

# Drop TransferPolicy's package docstring imports already handled; also drop
# ZeroActionPolicy dependency issues — keep as-is.
body = HEADER + "\n" + PID_EMBED + "\n\n" + PID_PLUS_EMBED + "\n\n" + TRANSFER_EMBED + "\n" + FOOTER
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(body, encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
