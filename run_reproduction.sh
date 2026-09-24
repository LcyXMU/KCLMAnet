#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PYTHON="/home/lcy/anaconda3/envs/Intelligentvehicle-gpu/bin/python"

if [[ -x "$CONDA_PYTHON" ]]; then
  PYTHON="$CONDA_PYTHON"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "$PACKAGE_DIR/reproduce_results.py" "$@"
