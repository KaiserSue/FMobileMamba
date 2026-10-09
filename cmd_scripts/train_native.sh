#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

python_bin="${PYTHON_BIN:-python}"
cfg_path="${CFG_PATH:-configs/mobilemamba/mobilemamba_t2.py}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_port="${MASTER_PORT:-29500}"
global_mode="${GLOBAL_MODE:-wt}"
local_mode="${LOCAL_MODE:-dwconv}"
seed="${SEED:-42}"
epoch_full="${EPOCH_FULL:-500}"
warmup_epochs="${WARMUP_EPOCHS:-30}"
test_start_epoch="${TEST_START_EPOCH:-400}"
batch_size="${BATCH_SIZE:-200}"
learning_rate="${LEARNING_RATE:-1.5e-4}"
weight_decay="${WEIGHT_DECAY:-0.05}"
nb_classes="${NB_CLASSES:-20}"
ft="${FT:-false}"
save_per_epoch="${SAVE_PER_EPOCH:-20}"
test_per_epoch="${TEST_PER_EPOCH:-5}"
num_workers_per_gpu="${NUM_WORKERS_PER_GPU:-8}"
checkpoint_root="${CHECKPOINT_ROOT:-runs/mobilemamba/500_epochs_mobilemamba}"
resume_dir="${RESUME_DIR:-}"
data_root="${DATA_ROOT:-}"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

if [[ ! -f "$cfg_path" ]]; then
    echo "Configuration file does not exist: $cfg_path" >&2
    exit 2
fi
if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python command is unavailable: $python_bin" >&2
    exit 2
fi
if [[ ! "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NPROC_PER_NODE must be a positive integer' >&2
    exit 2
fi
if [[ ! "$master_port" =~ ^[0-9]+$ ]] || (( 10#$master_port < 1 || 10#$master_port > 65535 )); then
    echo 'MASTER_PORT must be between 1 and 65535' >&2
    exit 2
fi
case "$global_mode" in fft|wt) ;; *) echo 'GLOBAL_MODE must be fft or wt' >&2; exit 2 ;; esac
case "$local_mode" in layeroperator|dwconv) ;; *) echo 'LOCAL_MODE must be layeroperator or dwconv' >&2; exit 2 ;; esac
ft="${ft,,}"
case "$ft" in true|false) ;; *) echo 'FT must be true or false' >&2; exit 2 ;; esac
for value in "$epoch_full" "$batch_size" "$nb_classes" "$save_per_epoch" "$test_per_epoch"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Positive integer training setting required' >&2; exit 2; }
done
for value in "$seed" "$warmup_epochs" "$test_start_epoch" "$num_workers_per_gpu"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Nonnegative integer training setting required' >&2; exit 2; }
done
[[ "$learning_rate" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || { echo 'LEARNING_RATE must be numeric' >&2; exit 2; }
[[ "$weight_decay" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || { echo 'WEIGHT_DECAY must be numeric' >&2; exit 2; }
[[ -n "$checkpoint_root" ]] || { echo 'CHECKPOINT_ROOT must not be empty' >&2; exit 2; }

opts=(
    "seed=$seed"
    "epoch_full=$epoch_full"
    "warmup_epochs=$warmup_epochs"
    "test_start_epoch=$test_start_epoch"
    "batch_size=$batch_size"
    "lr=$learning_rate"
    "weight_decay=$weight_decay"
    "nb_classes=$nb_classes"
    "ft=${ft^}"
    "trainer.epoch_full=$epoch_full"
    "trainer.scheduler_kwargs.warmup_epochs=$warmup_epochs"
    "trainer.scheduler_kwargs.lr_min=$(awk -v lr="$learning_rate" 'BEGIN { printf "%.12g", lr / 100 }')"
    "trainer.scheduler_kwargs.warmup_lr=$(awk -v lr="$learning_rate" 'BEGIN { printf "%.12g", lr / 1000 }')"
    "trainer.test_start_epoch=$test_start_epoch"
    "trainer.data.batch_size=$batch_size"
    "trainer.save_per_epoch=$save_per_epoch"
    "trainer.test_per_epoch=$test_per_epoch"
    "trainer.data.num_workers_per_gpu=$num_workers_per_gpu"
    "trainer.checkpoint=$checkpoint_root"
    "trainer.resume_dir=$resume_dir"
    "optim.lr=$learning_rate"
    "optim.optim_kwargs.weight_decay=$weight_decay"
    "data.nb_classes=$nb_classes"
    "model.model_kwargs.num_classes=$nb_classes"
    "trainer.mixup_kwargs.num_classes=$nb_classes"
    "model.model_kwargs.global_mode=$global_mode"
    "model.model_kwargs.local_mode=$local_mode"
)
if [[ -n "$data_root" ]]; then
    opts+=("data.root_dir=$data_root" "data.root=$data_root")
fi

command=("$python_bin")
if (( nproc_per_node > 1 )); then
    command+=(-m torch.distributed.run
        "--nproc_per_node=$nproc_per_node" --nnodes=1 "--master_port=$master_port")
fi
command+=(run.py -c "$cfg_path" -m train "${opts[@]}" "$@")

if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
