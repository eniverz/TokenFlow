#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/compute_norm.sh --task <task_name> [--num-workers N] [--batch-size N] [--output-root DIR]

Examples:
  bash scripts/compute_norm.sh --task tokenflow_stage2_v2_g05
  bash scripts/compute_norm.sh --task tokenflow_stage2_v2_g05 --num-workers 16 --batch-size 128
  bash scripts/compute_norm.sh --task tokenflow_stage2_v2_g05 --output-root /path/to/project/root

Options:
  --task <task_name>  Required. Hydra task config name under configs/task.
  --num-workers N    Data loading workers; default: 64.
  --batch-size N     Samples per DataLoader batch; default: 256.
  --output-root DIR  Base directory for relative norm paths; default: repository root.
  -h, --help         Show this help message.

Output:
  Writes action/state statistics to the paths configured by the task's normalizer.
  Each file contains global and stepwise statistics.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_NAME=""
NUM_WORKERS=64
BATCH_SIZE=256
OUTPUT_ROOT="$PROJECT_ROOT"

while (($# > 0)); do
    case "$1" in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
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

cd "$PROJECT_ROOT"

exec python -m resVLA.scripts.compute_norm \
    --task "$TASK_NAME" \
    --num-workers "$NUM_WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --output-root "$OUTPUT_ROOT"
