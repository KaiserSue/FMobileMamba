#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
: "${DATA_ROOT:?DATA_ROOT must point to the original dataset}"
conda_bin="${CONDA_BIN:-${CONDA_EXE:-conda}}"
args=(--data-root "$DATA_ROOT" --epochs "${EPOCHS:-1}" --seed "${SEED:-42}")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then args+=(--output-root "$OUTPUT_ROOT"); fi
if [[ "${KEEP_TEMP:-false}" == true ]]; then args+=(--keep-temp); fi
exec "$conda_bin" run --no-capture-output -n "${CONDA_ENV:-MobileMamba}" python -m tools.smoke_native_cv "${args[@]}"
