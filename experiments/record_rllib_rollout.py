from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.record_rollout import main as _record_rollout_main


if __name__ == "__main__":
    sys.argv[1:1] = ["--controller", "rllib"]
    _record_rollout_main()
