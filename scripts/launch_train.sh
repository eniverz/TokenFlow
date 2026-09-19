#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/launch_train.sh --task <task_name> --gpus <gpu_ids> [--port <port>] [-- <hydra overrides...>]

Examples:
  scripts/launch_train.sh --task resvla_libero --gpus 0,1,2,3
  WANDB_MODE=offline scripts/launch_train.sh --task resvla_libero --gpus 0,1
  scripts/launch_train.sh --task resvla_libero --gpus 0,1 -- num_train_steps=100000

Options:
  --task <task_name>  Hydra task config name.
  --gpus <gpu_ids>    Comma-separated GPU IDs, for example 0,1,2,3.
  --port <port>       First port to scan; default is 29500.
  -h, --help          Show this help message.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_NAME=""
GPU_LIST=""
BASE_PORT="29500"
HYDRA_OVERRIDES=()

while (($# > 0)); do
    case "$1" in
        --task)
            if (($# < 2)); then
                echo "error: --task requires a value" >&2
                usage >&2
                exit 2
            fi
            TASK_NAME="$2"
            shift 2
            ;;
        --gpus)
            if (($# < 2)); then
                echo "error: --gpus requires a value" >&2
                usage >&2
                exit 2
            fi
            GPU_LIST="$2"
            shift 2
            ;;
        --port)
            if (($# < 2)); then
                echo "error: --port requires a value" >&2
                usage >&2
                exit 2
            fi
            BASE_PORT="$2"
            shift 2
            ;;
        --)
            shift
            HYDRA_OVERRIDES+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$TASK_NAME" ]]; then
    echo "error: --task is required" >&2
    usage >&2
    exit 2
fi

if [[ -z "$GPU_LIST" ]]; then
    echo "error: --gpus is required" >&2
    usage >&2
    exit 2
fi

if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "error: --gpus must be comma-separated GPU IDs, for example 0,1,2,3" >&2
    exit 2
fi

if [[ ! "$BASE_PORT" =~ ^[0-9]+$ ]] || ((BASE_PORT < 1 || BASE_PORT > 65535)); then
    echo "error: --port must be an integer between 1 and 65535" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NUM_PROCESSES="${#GPU_IDS[@]}"

# Bash implements /dev/tcp as a redirection target.  This does not require
# lsof, nc, ss, netstat, or any other additional package.
port_is_busy() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

MAIN_PROCESS_PORT=""
for ((candidate_port=BASE_PORT; candidate_port<=65535; candidate_port++)); do
    if ! port_is_busy "$candidate_port"; then
        MAIN_PROCESS_PORT="$candidate_port"
        break
    fi
done

if [[ -z "$MAIN_PROCESS_PORT" ]]; then
    echo "error: no available TCP port found from $BASE_PORT through 65535" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

echo "Launching task=$TASK_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "num_machines=1 num_processes=$NUM_PROCESSES main_process_port=$MAIN_PROCESS_PORT"

export WANDB_MODE="${WANDB_MODE:-online}"

exec accelerate launch \
    --num_machines 1 \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m resVLA.scripts.train \
    "task=$TASK_NAME" \
    "${HYDRA_OVERRIDES[@]}"
