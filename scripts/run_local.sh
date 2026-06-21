#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python virtualenv not found at ${VENV_PYTHON}"
  echo "Create it first, then install requirements."
  exit 1
fi

cd "${ROOT_DIR}"
exec "${VENV_PYTHON}" -m uvicorn src.main:app --host "${HOST}" --port "${PORT}" --reload
