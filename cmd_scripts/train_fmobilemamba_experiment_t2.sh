#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

python_bin="${PYTHON_BIN:-python}"
cfg_path="${CFG_PATH:-configs/mobilemamba/mobilemamba_t2.py}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_port="${MASTER_PORT:-29500}"
global_mode="${GLOBAL_MODE:-fft}"
local_mode="${LOCAL_MODE:-dwconv}"
fft_kernel_size="${FFT_KERNEL_SIZE:-3}"
seed="${SEED:-42}"
epoch_full="${EPOCH_FULL:-500}"
warmup_epochs="${WARMUP_EPOCHS:-30}"
test_start_epoch="${TEST_START_EPOCH:-400}"
batch_size="${BATCH_SIZE:-200}"
learning_rate="${LEARNING_RATE:-1.5e-4}"
weight_decay="${WEIGHT_DECAY:-0.05}"
nb_classes="${NB_CLASSES:-200}"
ft="${FT:-false}"
save_per_epoch="${SAVE_PER_EPOCH:-20}"
test_per_epoch="${TEST_PER_EPOCH:-5}"
num_workers_per_gpu="${NUM_WORKERS_PER_GPU:-8}"
resume_dir="${RESUME_DIR:-}"
data_root="${DATA_ROOT:-}"
model_variant="${MODEL_VARIANT:-}"

if [[ -z "$model_variant" ]]; then
    cfg_filename="${cfg_path##*/}"
    cfg_stem="${cfg_filename%.py}"
    case "$cfg_stem" in
        mobilemamba_t2|mobilemamba_t4|mobilemamba_s6|mobilemamba_b1|mobilemamba_b2|mobilemamba_b4)
            model_variant="${cfg_stem#mobilemamba_}"
            ;;
        *)
            echo "Cannot infer model variant from CFG_PATH; set MODEL_VARIANT to T2, T4, S6, B1, B2, or B4" >&2
            exit 2
            ;;
    esac
fi
model_variant="${model_variant^^}"
case "$model_variant" in
    T2|T4|S6|B1|B2|B4) ;;
    *) echo 'MODEL_VARIANT must be T2, T4, S6, B1, B2, or B4' >&2; exit 2 ;;
esac
model_name="FMobileMamba_experiment_${model_variant}"
checkpoint_root="${CHECKPOINT_ROOT:-runs/mobilemamba/${model_name}_${fft_kernel_size}}"

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
[[ "${global_mode,,}" == fft ]] || { echo 'GLOBAL_MODE must be fft' >&2; exit 2; }
global_mode=fft
case "$local_mode" in layeroperator|dwconv) ;; *) echo 'LOCAL_MODE must be layeroperator or dwconv' >&2; exit 2 ;; esac
case "$fft_kernel_size" in 3|5|7|k) ;; *) echo 'FFT_KERNEL_SIZE must be 3, 5, 7, or k' >&2; exit 2 ;; esac
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
[[ "$checkpoint_root" != runs/mobilemamba/500_epochs_mobilemamba ]] || { echo 'CHECKPOINT_ROOT must differ from baseline' >&2; exit 2; }

for opt in "$@"; do
    if [[ "$opt" != *=* ]] || [[ "${opt%%=*}" != *.* && "${opt%%=*}" != nb_classes ]]; then
        echo "Expected path.key=value override: $opt" >&2
        exit 2
    fi
    key="${opt%%=*}"
    case "$key" in
        model|model.name|model.model_kwargs|model.model_kwargs.global_mode|model.model_kwargs.fft_kernel_size|nb_classes|data|data.nb_classes|trainer|trainer.mixup_kwargs|trainer.mixup_kwargs.num_classes|model.model_kwargs.num_classes)
            echo "Override is fixed for this model: $key" >&2
            exit 2
            ;;
    esac
done

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
    "model.name=$model_name"
    "model.model_kwargs.num_classes=$nb_classes"
    "model.model_kwargs.fft_kernel_size=$fft_kernel_size"
    "trainer.mixup_kwargs.num_classes=$nb_classes"
    "trainer.mixup_kwargs.mixup_alpha=0.5"
    "trainer.mixup_kwargs.cutmix_alpha=0.5"
    "trainer.mixup_kwargs.switch_prob=0.5"
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
