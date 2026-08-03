#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3.10 or 3.11 executable and retry." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit("Use Python 3.10 or 3.11 for the thesis server environment.")
print(f"Using Python {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[server]"

SUMO_HOME_VALUE="$(python - <<'PY'
import sumo

print(sumo.SUMO_HOME)
PY
)"
export SUMO_HOME="$SUMO_HOME_VALUE"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

python - <<'PY'
import shutil
import sys

import libsumo
import ray
import sumolib
import torch
import traci
import wandb

sumo_binary = shutil.which("sumo")
if not sumo_binary:
    raise SystemExit("SUMO binary not found on PATH after installing eclipse-sumo.")

print("Remote venv setup OK")
print(f"python={sys.executable}")
print(f"sumo={sumo_binary}")
print(f"sumolib={sumolib.__file__}")
print(f"traci={traci.__file__}")
print(f"libsumo={libsumo.__file__}")
print(f"ray={ray.__version__}")
print(f"torch={torch.__version__}")
print(f"wandb={wandb.__version__}")
PY

echo
echo "Activate this environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then set SUMO_HOME for each shell or job with:"
echo '  export SUMO_HOME="$(python -c '\''import sumo; print(sumo.SUMO_HOME)'\'')"'
echo '  export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"'
