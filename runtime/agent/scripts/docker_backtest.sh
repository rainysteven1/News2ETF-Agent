#!/usr/bin/env bash
set -euo pipefail

IMAGE="news2etf-agent"
MODE="backtest"
START_DATE=""
END_DATE=""
WEEK=""
RUN_ID=""
CONFIG_PATH=""
LOG_FILE=""
RESUME_FROM_WEEK=""
RESUME_TO_WEEK=""
RESUME_LATEST="0"
DIAGNOSE_PATH=""
START_WEEK=""
END_WEEK=""
MAX_ISSUES=""
SKIP_BUILD="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

collect_env_files() {
  local runtime_env_dir="${REPO_ROOT}/runtime/agent"
  local env_files

  env_files="$(find "${runtime_env_dir}" -maxdepth 1 -type f \
    \( -name '.env' -o -name '.env.*' \) \
    ! -name '.env.example' \
    ! -name '.env.sample' \
    | sort)"

  if [[ -n "${env_files}" ]]; then
    printf '%s\n' "${env_files}"
    return 0
  fi

  find "${REPO_ROOT}" -maxdepth 1 -type f \
    \( -name '.env' -o -name '.env.*' \) \
    ! -name '.env.example' \
    ! -name '.env.sample' \
    | sort
}

to_container_path() {
  local input="$1"
  if [[ -z "${input}" ]]; then
    return 0
  fi

  python - "$REPO_ROOT" "$input" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
raw = Path(sys.argv[2])
host_path = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
try:
    rel = host_path.relative_to(repo_root)
except ValueError:
    print(host_path)
else:
    print(Path("/app") / rel)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    decide)
      MODE="decide"
      shift
      ;;
    diagnose|--diagnose-backtest)
      MODE="diagnose-backtest"
      shift
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --week)
      WEEK="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --resume-from-week)
      RESUME_FROM_WEEK="$2"
      shift 2
      ;;
    --resume-to-week)
      RESUME_TO_WEEK="$2"
      shift 2
      ;;
    --resume-latest)
      RESUME_LATEST="1"
      shift
      ;;
    --path)
      DIAGNOSE_PATH="$2"
      shift 2
      ;;
    --start-week)
      START_WEEK="$2"
      shift 2
      ;;
    --end-week)
      END_WEEK="$2"
      shift 2
      ;;
    --max-issues)
      MAX_ISSUES="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${MODE}" == "backtest" ]]; then
  if [[ -z "${START_DATE}" || -z "${END_DATE}" ]]; then
    echo "--start-date and --end-date are required for backtest mode" >&2
    exit 2
  fi
elif [[ "${MODE}" == "decide" ]]; then
  if [[ -z "${WEEK}" ]]; then
    echo "--week is required for decide mode" >&2
    exit 2
  fi
elif [[ "${MODE}" == "diagnose-backtest" ]]; then
  if [[ -z "${RUN_ID}" && -z "${DIAGNOSE_PATH}" ]]; then
    echo "--run-id or --path is required for diagnose-backtest mode" >&2
    exit 2
  fi
else
  echo "Unsupported mode: ${MODE}" >&2
  exit 2
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  docker build --network host -f "${REPO_ROOT}/runtime/agent/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
fi

if [[ "${MODE}" == "backtest" ]]; then
  cmd=(python main.py backtest --start-date "${START_DATE}" --end-date "${END_DATE}")
elif [[ "${MODE}" == "decide" ]]; then
  cmd=(python main.py decide --week "${WEEK}")
else
  cmd=(python main.py diagnose-backtest)
fi

env_args=()
while IFS= read -r env_file; do
  [[ -n "${env_file}" ]] || continue
  env_args+=(--env-file "${env_file}")
done < <(collect_env_files)

if [[ -n "${CONFIG_PATH}" ]]; then
  cmd+=(--config "$(to_container_path "${CONFIG_PATH}")")
fi
if [[ ("${MODE}" == "backtest" || "${MODE}" == "decide") && -n "${LOG_FILE}" ]]; then
  cmd+=(--log-file "$(to_container_path "${LOG_FILE}")")
fi
if [[ -n "${RUN_ID}" ]]; then
  cmd+=(--run-id "${RUN_ID}")
fi
if [[ "${MODE}" == "backtest" && "${RESUME_LATEST}" == "1" ]]; then
  cmd+=(--resume-latest)
fi
if [[ "${MODE}" == "backtest" && -n "${RESUME_FROM_WEEK}" ]]; then
  cmd+=(--resume-from-week "${RESUME_FROM_WEEK}")
fi
if [[ "${MODE}" == "backtest" && -n "${RESUME_TO_WEEK}" ]]; then
  cmd+=(--resume-to-week "${RESUME_TO_WEEK}")
fi
if [[ "${MODE}" == "diagnose-backtest" && -n "${DIAGNOSE_PATH}" ]]; then
  cmd+=(--path "$(to_container_path "${DIAGNOSE_PATH}")")
fi
if [[ "${MODE}" == "diagnose-backtest" && -n "${START_WEEK}" ]]; then
  cmd+=(--start-week "${START_WEEK}")
fi
if [[ "${MODE}" == "diagnose-backtest" && -n "${END_WEEK}" ]]; then
  cmd+=(--end-week "${END_WEEK}")
fi
if [[ "${MODE}" == "diagnose-backtest" && -n "${MAX_ISSUES}" ]]; then
  cmd+=(--max-issues "${MAX_ISSUES}")
fi

container_id="$(
  docker run -d --rm --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  "${env_args[@]}" \
  -e HOME=/tmp/news2etf-home \
  -e XDG_CACHE_HOME=/tmp/news2etf-cache \
  -e NEWS2ETF_REPO_ROOT=/app \
  -e NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
  -e NEWS2ETF_SHARED_DATA_ROOT=/app/data \
  -v "${REPO_ROOT}/data:/app/data" \
  -v "${REPO_ROOT}/runtime/agent/models:/app/runtime/agent/models" \
  -v "${REPO_ROOT}/runtime/agent/data/inputs:/app/runtime/agent/data/inputs" \
  -v "${REPO_ROOT}/runtime/agent:/app/runtime/agent" \
  "${IMAGE}" \
  "${cmd[@]}"
)"

echo "Started container: ${container_id}"
echo "Follow logs with: docker logs -f ${container_id}"
