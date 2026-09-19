#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_NAME="LIBERO"
DEFAULT_SUITES="libero_spatial,libero_object,libero_goal,libero_10"

CHECKPOINT_DIR=""
TASK_NAME=""
GPU_SPEC=""
SUITE_SPEC="${DEFAULT_SUITES}"
BASE_PORT=10093
DTYPE="bf16"
EXECUTION_HORIZON=0
EXTRA_EVAL_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  run_parallel_eval.sh \
    --task resvla_libero \
    --checkpoint-dir results/<run_dir>/checkpoint/step_<N> \
    [--gpus 0,1,2,3] [--base-port 10093] [--dtype bf16] \
    [--execution-horizon 8] \
    [--suites libero_spatial,libero_object,libero_goal,libero_10] \
    [-- <extra eval arguments>]

Required entries in the repository root .env:
  RESVLA_PYTHON=/path/to/resvla/env/bin/python
  LIBERO_PYTHON=/path/to/libero/env/bin/python

GPU selection defaults to CUDA_VISIBLE_DEVICES, then falls back to nvidia-smi.
Suites are assigned to detected GPUs round-robin.
Each invocation writes to LIBERO/<timestamp>/<suite> under the checkpoint run directory.
EOF
}

while (($#)); do
    case "$1" in
        --task) TASK_NAME="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --gpus) GPU_SPEC="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --execution-horizon) EXECUTION_HORIZON="$2"; shift 2 ;;
        --suites) SUITE_SPEC="$2"; shift 2 ;;
        --) shift; EXTRA_EVAL_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    exit 2
fi
set -a
source "${ENV_FILE}"
set +a
RESVLA_PYTHON="${RESVLA_PYTHON:-}"
BENCHMARK_PYTHON="${LIBERO_PYTHON:-}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    usage >&2
    exit 2
fi
if [[ -z "${TASK_NAME}" ]]; then
    echo "--task is required to load the training Hydra config." >&2
    usage >&2
    exit 2
fi
if [[ ! -x "${RESVLA_PYTHON}" ]]; then
    echo "resVLA Python is not executable: ${RESVLA_PYTHON}" >&2
    exit 2
fi
if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
    echo "Benchmark Python is not executable: ${BENCHMARK_PYTHON}" >&2
    exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    exit 2
fi

CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
CHECKPOINT_PARENT="$(dirname "${CHECKPOINT_DIR}")"
RUN_DIR="$(dirname "${CHECKPOINT_PARENT}")"
RUN_ID="$(basename "${RUN_DIR}")"

GPUS=()
if [[ -n "${GPU_SPEC}" ]]; then
    IFS=',' read -r -a GPUS <<< "${GPU_SPEC}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r gpu; do
        [[ -n "${gpu}" ]] && GPUS+=("${gpu}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

for i in "${!GPUS[@]}"; do
    GPUS[$i]="${GPUS[$i]//[[:space:]]/}"
done
if ((${#GPUS[@]} == 0)); then
    echo "No GPU detected. Set --gpus or CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi

IFS=',' read -r -a SUITES <<< "${SUITE_SPEC}"
if ((${#SUITES[@]} == 0)); then
    echo "No suites configured." >&2
    exit 2
fi

EVAL_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_BASE_DIR="${RUN_DIR}/${BENCHMARK_NAME}"
mkdir -p "${EVAL_BASE_DIR}"
EVAL_ROOT=""
for ((suffix=0; suffix<100; suffix++)); do
    if ((suffix == 0)); then
        eval_id="${EVAL_TIMESTAMP}"
    else
        eval_id="${EVAL_TIMESTAMP}_${suffix}"
    fi
    candidate="${EVAL_BASE_DIR}/${eval_id}"
    if mkdir "${candidate}" 2>/dev/null; then
        EVAL_ID="${eval_id}"
        EVAL_ROOT="${candidate}"
        break
    fi
done
if [[ -z "${EVAL_ROOT}" ]]; then
    echo "Failed to create a unique evaluation directory under ${EVAL_BASE_DIR}" >&2
    exit 2
fi
echo "Evaluation round: ${EVAL_ROOT}"

SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
    echo "Starting cleanup..."
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
    for pid in "${EVAL_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${EVAL_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
for i in "${!SUITES[@]}"; do
    suite="${SUITES[$i]//[[:space:]]/}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    port=$((BASE_PORT + i))
    suite_dir="${EVAL_ROOT}/${suite}"
    mkdir -p "${suite_dir}"

    server_log="${suite_dir}/server.log"
    eval_log="${suite_dir}/eval.log"
    progress_file="${suite_dir}/progress.jsonl"
    benchmark_config_file="${suite_dir}/benchmark_config.json"

    echo "Starting ${suite}: GPU=${gpu}, port=${port}, logs=${suite_dir}"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" "${RESVLA_PYTHON}" -m resVLA.deployment.server \
        "task=${TASK_NAME}" \
        "+checkpoint_dir=${CHECKPOINT_DIR}" \
        "+device=cuda" \
        "+dtype=${DTYPE}" \
        "+port=${port}" \
        >"${server_log}" 2>&1 &
    SERVER_PIDS+=("$!")

    PYTHONUNBUFFERED=1 "${BENCHMARK_PYTHON}" "${SCRIPT_DIR}/eval_libero_ws.py" \
        --host 127.0.0.1 \
        --port "${port}" \
        --execution-horizon "${EXECUTION_HORIZON}" \
        --task-suite-names "${suite}" \
        --run-name "${RUN_ID}_${EVAL_ID}_${suite}" \
        --video-out-path "${suite_dir}/videos" \
        --result-file "${suite_dir}/result.json" \
        --progress-file "${progress_file}" \
        --benchmark-config-file "${benchmark_config_file}" \
        "${EXTRA_EVAL_ARGS[@]}" \
        >"${eval_log}" 2>&1 &
    EVAL_PIDS+=("$!")
done

status=0
for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
