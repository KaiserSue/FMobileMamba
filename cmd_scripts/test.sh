#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-MobileMamba}"
MODEL_SCALE="${MODEL_SCALE-t2}"
LOG_DIR="${LOG_DIR:-evals/fmobilemaba_t2_eval_new/test_eval}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOGGER_RANK="${LOGGER_RANK:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-192}"
NB_CLASSES="${NB_CLASSES:-20}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-125}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-8}"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-conda}}"

die() {
    echo "$1" >&2
    exit 2
}

resolve_model_defaults() {
    local model_scale="$1"

    case "$model_scale" in
        b1)
            DEFAULT_CFG_PATH="configs/mobilemamba/mobilemamba_b1.py"
            DEFAULT_CHECKPOINT_PATH="runs/mobilemamba/500_epochs_fmobilemamba_b1_new/fast_pipeline/run_1789430177731391856/FastCLSTrainer_FMobileMamba_B1_DefaultCLS_20260915-075617/best_epoch420.pth"
            DEFAULT_GLOBAL_MODE="wt"
            DEFAULT_LOCAL_MODE="dwconv"
            ;;
        t2)
            DEFAULT_CFG_PATH="configs/mobilemamba/mobilemamba_t2.py"
            DEFAULT_CHECKPOINT_PATH="runs/mobilemamba/500_epochs_fmobilemamba_t2_new/fast_pipeline/run_1789695064583429771/FastCLSTrainer_FMobileMamba_T2_DefaultCLS_20260918-093104/best_epoch515.pth"
            DEFAULT_GLOBAL_MODE="fft"
            DEFAULT_LOCAL_MODE="layeroperator"
            ;;
        *)
            die 'MODEL_SCALE must be b1 or t2'
            ;;
    esac
}

validate_launcher_params() {
    case "$MODEL_SCALE" in b1|t2) ;; *) die 'MODEL_SCALE must be b1 or t2' ;; esac
    [[ -n "$CHECKPOINT_PATH" ]] || die 'CHECKPOINT_PATH must not be empty'
    [[ -n "$LOG_DIR" ]] || die 'LOG_DIR must not be empty'
    [[ -n "$CFG_PATH" ]] || die 'CFG_PATH must not be empty'
    [[ -n "$CONDA_ENV" ]] || die 'CONDA_ENV must not be empty'
    [[ -n "$CONDA_BIN" ]] || die 'CONDA_BIN must not be empty'
    [[ -f "$CHECKPOINT_PATH" && -r "$CHECKPOINT_PATH" ]] || die 'CHECKPOINT_PATH must identify a readable file'
    [[ -f "$CFG_PATH" && -r "$CFG_PATH" ]] || die 'CFG_PATH must identify a readable file'
    case "$GLOBAL_MODE" in fft|wt) ;; *) die 'GLOBAL_MODE must be fft or wt' ;; esac
    case "$LOCAL_MODE" in layeroperator|dwconv) ;; *) die 'LOCAL_MODE must be layeroperator or dwconv' ;; esac

    local name value
    for name in NPROC_PER_NODE IMAGE_SIZE NB_CLASSES BATCH_SIZE_PER_GPU; do
        value="${!name}"
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
    done
    for name in LOGGER_RANK NUM_WORKERS_PER_GPU; do
        value="${!name}"
        [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a nonnegative integer"
    done
    (( 10#$LOGGER_RANK < 10#$NPROC_PER_NODE )) || die 'LOGGER_RANK must be smaller than NPROC_PER_NODE'
    [[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] || die 'MASTER_PORT must be between 1 and 65535'
    (( 10#$MASTER_PORT >= 1 && 10#$MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be between 1 and 65535'
}

if (($#)); then
    die 'test.sh does not accept command-line arguments; use its named environment variables'
fi

resolve_model_defaults "$MODEL_SCALE"
GLOBAL_MODE="${GLOBAL_MODE:-$DEFAULT_GLOBAL_MODE}"
LOCAL_MODE="${LOCAL_MODE:-$DEFAULT_LOCAL_MODE}"
CFG_PATH="${CFG_PATH-$DEFAULT_CFG_PATH}"
CHECKPOINT_PATH="${CHECKPOINT_PATH-$DEFAULT_CHECKPOINT_PATH}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
validate_launcher_params

command=(
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV"
    python -m torch.distributed.run
    "--nproc_per_node=$NPROC_PER_NODE"
    --nnodes=1
    "--master_port=$MASTER_PORT"
    -m evals.native_test
    --cfg-path "$CFG_PATH"
    --checkpoint-path "$CHECKPOINT_PATH"
    --log-dir "$LOG_DIR"
    --global-mode "$GLOBAL_MODE"
    --local-mode "$LOCAL_MODE"
    --image-size "$IMAGE_SIZE"
    --nb-classes "$NB_CLASSES"
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU"
    --num-workers-per-gpu "$NUM_WORKERS_PER_GPU"
    --dist-url env://
    --logger-rank "$LOGGER_RANK"
)

exec "${command[@]}"
