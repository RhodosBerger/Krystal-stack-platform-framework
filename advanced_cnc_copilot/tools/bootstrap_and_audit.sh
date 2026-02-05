#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_ENV_DIR="${ROOT_DIR}/.venv"

log() {
  echo "[bootstrap] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[bootstrap][error] required command not found: $1" >&2
    exit 1
  fi
}

setup_python() {
  require_cmd python3
  log "Creating venv at ${PY_ENV_DIR}"
  python3 -m venv "${PY_ENV_DIR}"
  # shellcheck disable=SC1091
  source "${PY_ENV_DIR}/bin/activate"
  log "Upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel

  if [[ -f "${ROOT_DIR}/flask_service/requirements.txt" ]]; then
    log "Installing flask service requirements"
    python -m pip install -r "${ROOT_DIR}/flask_service/requirements.txt"
  fi

  if [[ -f "${ROOT_DIR}/dist/requirements.txt" ]]; then
    log "Installing dist requirements"
    python -m pip install -r "${ROOT_DIR}/dist/requirements.txt"
  fi

  log "Running pip check"
  python -m pip check || true
}

setup_node_project() {
  local dir="$1"
  if [[ -f "${dir}/package.json" ]]; then
    require_cmd npm
    log "Installing npm deps in ${dir}"
    (cd "${dir}" && npm install)
    log "Dependency tree (${dir})"
    (cd "${dir}" && npm ls --depth=0 || true)
  fi
}

print_next_steps() {
  cat <<MSG

[bootstrap] Done.
Next steps:
  1) Activate python env: source ${PY_ENV_DIR}/bin/activate
  2) Run backend: (cd ${ROOT_DIR} && uvicorn backend.main:app --reload)
  3) Open dashboard: http://localhost:8000/dashboard/hub.html
  4) Optional machine view: http://localhost:8000/dashboard/index.html?machine_id=CNC-001
MSG
}

main() {
  setup_python
  setup_node_project "${ROOT_DIR}/frontend-react"
  setup_node_project "${ROOT_DIR}/frontend-vue"
  setup_node_project "${ROOT_DIR}/cms/dashboard/pregeneratedby me"
  print_next_steps
}

main "$@"
