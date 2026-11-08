#!/usr/bin/env bash
# Start FRAM3. First run creates a Python environment in .venv.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.13 python3; do command -v "$c" >/dev/null && { PY=$c; break; }; done
fi
if [ ! -x .venv/bin/python ]; then
  echo "FRAM3: eerste keer opstarten, Python-omgeving maken met $PY ..."
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt || .venv/bin/pip install $(grep -vE '^(#|cascadio)' requirements.txt | sed 's/ ;.*//')
fi
exec .venv/bin/python -m fram3 "$@"
