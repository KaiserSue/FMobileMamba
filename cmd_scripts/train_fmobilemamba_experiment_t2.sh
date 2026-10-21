#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
started_at="$(date +%m%d%H%M)"

python_bin="${PYTHON_BIN:-python}"
cfg_path="${CFG_PATH:-configs/mobilemamba/mobilemamba_b1.py}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_port="${MASTER_PORT:-29500}"
global_mode="${GLOBAL_MODE:-fft}"
local_mode="${LOCAL_MODE:-layeroperator}"
local_kernel_set="${LOCAL_KERNEL_SET:-reduced}"
force_dcnv3_rebuild="${FORCE_DCNV3_REBUILD:-false}"
fft_kernel_size="${FFT_KERNEL_SIZE:-3}"
seed="${SEED:-42}"
epoch_full="${EPOCH_FULL:-300}"
warmup_epochs="${WARMUP_EPOCHS:-30}"
test_start_epoch="${TEST_START_EPOCH:-200}"
batch_size="${BATCH_SIZE:-200}"
learning_rate="${LEARNING_RATE:-1.5e-4}"
weight_decay="${WEIGHT_DECAY:-0.05}"
nb_classes="${NB_CLASSES:-200}"
ft="${FT:-false}"
save_per_epoch="${SAVE_PER_EPOCH:-20}"
test_per_epoch="${TEST_PER_EPOCH:-5}"
num_workers_per_gpu="${NUM_WORKERS_PER_GPU:-8}"
resume_dir="${RESUME_DIR:-}"
data_root="${DATA_ROOT:-/home/apulis-dev/adhub/Tiny_ImageNet/0.1/tiny-imagenet-200}"
checkpoint_root="${CHECKPOINT_ROOT:-runs/tiny_imagenet/1}"

resolve_experiment_model_name() {
    local config_path="$1"
    if "$python_bin" - "$config_path" <<'PY'
import sys
from argparse import Namespace
from contextlib import redirect_stdout

# Configuration imports may print diagnostics; stdout carries only the model name.
with redirect_stdout(sys.stderr):
    from configs import get_cfg

    cfg = get_cfg(Namespace(
        cfg_path=sys.argv[1], mode="train", sleep=-1, memory=-1,
        dist_url="env://", logger_rank=0, opts=[],
    ))
    source_model_name = cfg.model.name

if not isinstance(source_model_name, str) or not source_model_name:
    raise SystemExit("model.name must be a nonempty string")

model_names = {
    f"{prefix}_{variant}": f"FMobileMamba_experiment_{variant}"
    for prefix in ("MobileMamba", "FMobileMamba", "FMobileMamba_experiment")
    for variant in ("T2", "T4", "S6", "B1", "B2", "B4")
}
if source_model_name not in model_names:
    raise SystemExit(f"Unsupported model.name: {source_model_name!r}")
print(model_names[source_model_name])
PY
    then
        return 0
    fi
    echo "Cannot resolve model.name from configuration: $config_path" >&2
    return 2
}

build_experiment_logdir_name() {
    local epoch_full="$1"
    local global_mode="$2"
    local local_mode="$3"
    local fft_kernel_size="$4"
    local started_at="$5"
    local name="${epoch_full}epochs_${global_mode}_${local_mode}"
    if [[ "$global_mode" == fft ]]; then
        name+="_${fft_kernel_size}"
    fi
    printf '%s_%s\n' "$name" "$started_at"
}

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
global_mode="${global_mode,,}"
case "$global_mode" in fft|wt) ;; *) echo 'GLOBAL_MODE must be fft or wt' >&2; exit 2 ;; esac
case "$local_mode" in layeroperator|dwconv|local_branch_new) ;; *) echo 'LOCAL_MODE must be layeroperator, dwconv, or local_branch_new' >&2; exit 2 ;; esac
case "$local_kernel_set" in original|reduced) ;; *) echo 'LOCAL_KERNEL_SET must be original or reduced' >&2; exit 2 ;; esac
force_dcnv3_rebuild="${force_dcnv3_rebuild,,}"
case "$force_dcnv3_rebuild" in true|false) ;; *) echo 'FORCE_DCNV3_REBUILD must be true or false' >&2; exit 2 ;; esac
if [[ "$global_mode" == fft ]]; then
    case "$fft_kernel_size" in 3|5|7|k|k1|k2) ;; *) echo 'FFT_KERNEL_SIZE must be 3, 5, 7, k, k1, or k2' >&2; exit 2 ;; esac
fi
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

effective_epoch_full="$epoch_full"
for opt in "$@"; do
    if [[ "$opt" != *=* ]] || [[ "${opt%%=*}" != *.* && "${opt%%=*}" != nb_classes ]]; then
        echo "Expected path.key=value override: $opt" >&2
        exit 2
    fi
    key="${opt%%=*}"
    case "$key" in
        model|model.name|model.model_kwargs|model.model_kwargs.global_mode|model.model_kwargs.local_mode|model.model_kwargs.local_kernel_set|model.model_kwargs.fft_kernel_size|nb_classes|data|data.nb_classes|trainer|trainer.logdir_name|trainer.mixup_kwargs|trainer.mixup_kwargs.num_classes|model.model_kwargs.num_classes)
            echo "Override is fixed for this model: $key" >&2
            exit 2
            ;;
    esac
    if [[ "$key" == trainer.epoch_full ]]; then
        effective_epoch_full="${opt#*=}"
    fi
done
[[ "$effective_epoch_full" =~ ^[1-9][0-9]*$ ]] || { echo 'trainer.epoch_full must be a positive integer' >&2; exit 2; }

if ! model_name="$(resolve_experiment_model_name "$cfg_path")"; then
    exit 2
fi

logdir_name="$(build_experiment_logdir_name \
    "$effective_epoch_full" "$global_mode" "$local_mode" "$fft_kernel_size" "$started_at")"
[[ -n "$checkpoint_root" ]] || { echo 'CHECKPOINT_ROOT must not be empty' >&2; exit 2; }
[[ "$checkpoint_root" != runs/mobilemamba/500_epochs_mobilemamba ]] || { echo 'CHECKPOINT_ROOT must differ from baseline' >&2; exit 2; }

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
    "trainer.logdir_name=$logdir_name"
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
if [[ "$local_mode" == local_branch_new ]]; then
    opts+=("model.model_kwargs.local_kernel_set=$local_kernel_set")
fi
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

check_dcnv3_health() {
    "$python_bin" - <<'PY'
import importlib.metadata

import torch
import DCNv3
from model.ops_dcnv3.modules import DCNv3 as DCNv3Module

distribution = importlib.metadata.distribution("DCNv3")
if distribution.version != "1.1":
    raise RuntimeError(
        f"expected DCNv3 distribution version 1.1, found {distribution.version}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable to PyTorch")

device = torch.device("cuda", torch.cuda.current_device())
module = DCNv3Module(
    channels=16,
    kernel_size=3,
    dw_kernel_size=3,
    stride=1,
    pad=1,
    dilation=1,
    group=16,
).to(device)
input_tensor = torch.randn(1, 8, 8, 16, device=device)
with torch.no_grad():
    output = module(input_tensor)
if tuple(output.shape) != (1, 8, 8, 16):
    raise RuntimeError(f"unexpected DCNv3 output shape: {tuple(output.shape)}")
print("DCNv3 CUDA health check passed with output shape [1, 8, 8, 16]")
PY
}

collect_dcnv3_prerequisite_errors() {
    local has_errors=false
    if ! "$python_bin" - <<'PY'
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys

required_packages = (
    "torch",
    "torchvision",
    "setuptools",
    "wheel",
    "ninja",
    "packaging",
    "pip",
)
errors = []
missing_packages = []
loaded_packages = {}
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)
        continue
    try:
        loaded_packages[package] = importlib.import_module(package)
    except Exception as error:
        errors.append(
            f"[Python packages] cannot import {package}: "
            f"{type(error).__name__}: {error}"
        )
if missing_packages:
    errors.append(
        "[Python packages] missing: " + ", ".join(missing_packages)
    )

torch = loaded_packages.get("torch")
cuda_home = None
if torch is not None:
    if torch.version.cuda is None:
        errors.append("[CUDA] PyTorch was built without CUDA support")
    if not torch.cuda.is_available():
        errors.append("[CUDA] torch.cuda.is_available() is false")
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_home = CUDA_HOME
    except Exception as error:
        errors.append(
            "[Python packages] cannot import torch CUDA build utilities: "
            f"{type(error).__name__}: {error}"
        )
    if cuda_home is None:
        errors.append("[CUDA] torch.utils.cpp_extension.CUDA_HOME is unset")

nvcc_path = shutil.which("nvcc")
if nvcc_path is None:
    errors.append("[CUDA] nvcc is unavailable on PATH")
if cuda_home is not None:
    cuda_home_nvcc = str(cuda_home) + "/bin/nvcc"
    if not os.path.isfile(cuda_home_nvcc) or not os.access(cuda_home_nvcc, os.X_OK):
        errors.append(f"[CUDA] nvcc is unavailable at {cuda_home_nvcc}")
if nvcc_path is not None and torch is not None:
    completed = subprocess.run(
        [nvcc_path, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode == 0:
        match = re.search(r"release\s+(\d+\.\d+)", completed.stdout)
        if match is None:
            errors.append("[CUDA] could not parse nvcc version")
        elif torch.version.cuda is not None:
            torch_cuda = ".".join(torch.version.cuda.split(".")[:2])
            if match.group(1) != torch_cuda:
                errors.append(
                    "[CUDA] PyTorch CUDA " + torch_cuda
                    + " does not match nvcc " + match.group(1)
                )
    else:
        errors.append(f"[CUDA] nvcc failed at {nvcc_path}")

for error in errors:
    print(error, file=sys.stderr)
raise SystemExit(1 if errors else 0)
PY
    then
        has_errors=true
    fi

    local tool
    for tool in gcc g++ flock; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "[Build tools] missing: $tool" >&2
            has_errors=true
        fi
    done

    local dcnv3_dir="$script_dir/../model/ops_dcnv3"
    if [[ ! -f "$dcnv3_dir/setup.py" ]]; then
        echo "[Source files] missing: $dcnv3_dir/setup.py" >&2
        has_errors=true
    fi
    if [[ ! -d "$dcnv3_dir/src" || ! -d "$dcnv3_dir/src/cuda" ]]; then
        echo "[Source files] missing DCNv3 src or src/cuda directory" >&2
        has_errors=true
    fi
    if ! compgen -G "$dcnv3_dir/src/*.cpp" >/dev/null \
            || ! compgen -G "$dcnv3_dir/src/cuda/*.cu" >/dev/null; then
        echo "[Source files] missing DCNv3 C++ or CUDA sources" >&2
        has_errors=true
    fi

    [[ "$has_errors" == false ]]
}

ensure_dcnv3_ready() {
    local initial_health=false
    if check_dcnv3_health; then
        initial_health=true
    fi
    if [[ "$initial_health" == true && "$force_dcnv3_rebuild" == false ]]; then
        echo "Reusing healthy CUDA DCNv3 installation"
        return
    fi

    if ! collect_dcnv3_prerequisite_errors; then
        echo "DCNv3 prerequisites are incomplete; training was not started" >&2
        return 1
    fi

    local resolved_python
    resolved_python="$("$python_bin" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    local environment_id
    environment_id="$(printf '%s' "$resolved_python" | cksum | awk '{print $1}')"
    local lock_file="/tmp/fmobilemamba_dcnv3_$(id -u)_${environment_id}.lock"
    exec 9>"$lock_file"
    flock 9

    if [[ "$force_dcnv3_rebuild" == false ]] && check_dcnv3_health; then
        echo "Reusing CUDA DCNv3 installed while waiting for the build lock"
        flock -u 9
        exec 9>&-
        return
    fi

    local dcnv3_dir="$script_dir/../model/ops_dcnv3"
    echo "Building CUDA DCNv3 from $dcnv3_dir with $resolved_python"
    if ! "$python_bin" -m pip install --no-build-isolation --no-deps \
            --force-reinstall --verbose "$dcnv3_dir"; then
        echo "DCNv3 build and installation failed; training was not started" >&2
        flock -u 9
        exec 9>&-
        return 1
    fi
    if ! check_dcnv3_health; then
        echo "DCNv3 post-install CUDA health check failed; training was not started" >&2
        flock -u 9
        exec 9>&-
        return 1
    fi
    flock -u 9
    exec 9>&-
}

if [[ "$local_mode" == local_branch_new ]]; then
    ensure_dcnv3_ready
fi

exec "${command[@]}"
