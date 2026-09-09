#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-MobileMamba}"
GLOBAL_MODE="${GLOBAL_MODE:-wt}"
LOCAL_MODE="${LOCAL_MODE:-dwconv}"
CFG_PATH="${CFG_PATH:-configs/mobilemamba/mobilemamba_t2.py}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-42}"
IMAGE_SIZE="${IMAGE_SIZE:-192}"
EPOCH_FULL="${EPOCH_FULL:-500}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
TEST_START_EPOCH="${TEST_START_EPOCH:-400}"
BATCH_SIZE="${BATCH_SIZE:-200}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NB_CLASSES="${NB_CLASSES:-20}"
FT="${FT:-false}"
SAVE_PER_EPOCH="${SAVE_PER_EPOCH:-20}"
TEST_PER_EPOCH="${TEST_PER_EPOCH:-5}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-runs/mobilemamba/500_epochs_origin_mobilemamba/fast_pipeline}"
RESUME_DIR="${RESUME_DIR:-}"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-conda}}"

die() {
    echo "$1" >&2
    exit 2
}

validate_launcher_params() {
    case "$GLOBAL_MODE" in fft|wt) ;; *) die 'GLOBAL_MODE must be fft or wt' ;; esac
    case "$LOCAL_MODE" in layeroperator|dwconv) ;; *) die 'LOCAL_MODE must be layeroperator or dwconv' ;; esac
    FT="${FT,,}"
    case "$FT" in true|false) ;; *) die 'FT must be true or false' ;; esac

    local name value
    for name in NPROC_PER_NODE IMAGE_SIZE EPOCH_FULL BATCH_SIZE NB_CLASSES SAVE_PER_EPOCH TEST_PER_EPOCH; do
        value="${!name}"
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
    done
    for name in SEED WARMUP_EPOCHS TEST_START_EPOCH NUM_WORKERS_PER_GPU; do
        value="${!name}"
        [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a nonnegative integer"
    done
    [[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] || die 'MASTER_PORT must be between 1 and 65535'
    (( 10#$MASTER_PORT >= 1 && 10#$MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be between 1 and 65535'
    [[ "$LEARNING_RATE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || die 'LEARNING_RATE must be a positive number'
    [[ ! "$LEARNING_RATE" =~ ^0*([.]0*)?([eE][+-]?[0-9]+)?$ ]] || die 'LEARNING_RATE must be a positive number'
    [[ "$WEIGHT_DECAY" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || die 'WEIGHT_DECAY must be a nonnegative number'
    [[ -n "$CHECKPOINT_ROOT" ]] || die 'CHECKPOINT_ROOT must not be empty'
}

build_config_overrides() {
    if "$SYNTHETIC_SMOKE"; then
        script_opts=(
            "model.model_kwargs.global_mode=$GLOBAL_MODE"
            "model.model_kwargs.local_mode=$LOCAL_MODE"
        )
        return
    fi
    script_opts=(
        "shared.seed=$SEED"
        "shared.size=$IMAGE_SIZE"
        "shared.epoch_full=$EPOCH_FULL"
        "shared.warmup_epochs=$WARMUP_EPOCHS"
        "shared.test_start_epoch=$TEST_START_EPOCH"
        "shared.batch_size=$BATCH_SIZE"
        "shared.lr=$LEARNING_RATE"
        "shared.weight_decay=$WEIGHT_DECAY"
        "shared.nb_classes=$NB_CLASSES"
        "shared.ft=$FT"
        "trainer.save_per_epoch=$SAVE_PER_EPOCH"
        "trainer.test_per_epoch=$TEST_PER_EPOCH"
        "trainer.data.num_workers_per_gpu=$NUM_WORKERS_PER_GPU"
        "trainer.checkpoint=$CHECKPOINT_ROOT"
        "trainer.resume_dir=$RESUME_DIR"
        "model.model_kwargs.global_mode=$GLOBAL_MODE"
        "model.model_kwargs.local_mode=$LOCAL_MODE"
    )
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
DRY_RUN=false
SYNTHETIC_SMOKE=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi
cli=()
opts=()
while (($#)); do
    case "$1" in
        -c|--cfg_path)
            (($# >= 2)) || { echo 'Missing configuration path' >&2; exit 2; }
            CFG_PATH="$2"; shift 2 ;;
        --sleep|--memory|--dist_url|--logger_rank)
            (($# >= 2)) || { echo 'Missing option value' >&2; exit 2; }
            cli+=("$1" "$2"); shift 2 ;;
        --synthetic-smoke) SYNTHETIC_SMOKE=true; cli+=("$1"); shift ;;
        --help|-h) cli+=("$1"); shift ;;
        -m|--mode) echo 'train.sh starts train; use run_fast.py for other modes' >&2; exit 2 ;;
        --*) echo "Unsupported CLI option: $1" >&2; exit 2 ;;
        *=*) opts+=("$1"); shift ;;
        *) echo "Expected CLI option or key=value: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$CFG_PATH" && -f run_fast.py ]] || { echo "Missing config or run_fast.py: $CFG_PATH" >&2; exit 2; }
validate_launcher_params
script_opts=()
build_config_overrides
command=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV"
    python -m torch.distributed.run "--nproc_per_node=$NPROC_PER_NODE"
    --nnodes=1 "--master_port=$MASTER_PORT" run_fast.py -c "$CFG_PATH" -m train
    "${cli[@]}" "${script_opts[@]}" "${opts[@]}")
if "$DRY_RUN"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
