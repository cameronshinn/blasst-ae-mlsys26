#!/bin/bash

# Exit on first error
set -e

# Change directory to the workspace root where the script is located
cd "$(dirname "$0")"

echo "Running flashinfer tests..."

cd external/flashinfer

# We must ensure that we use the local flashinfer rather than any pre-installed container version
python3 -m pip uninstall -y flashinfer --break-system-packages || true

# Install from source as per the README
python3 -m pip install --upgrade pip setuptools --break-system-packages
python3 -m pip install -v . --break-system-packages

# Point FLASHINFER_WORKSPACE_BASE to the local submodule so the pre-built cached kernels are used automatically
export FLASHINFER_WORKSPACE_BASE="$PWD"

# Run pytest on the specified file, passing along any extra arguments provided
# The skips_softmax parameter is the first one in the list, so True appears as "[True-"
python3 -m pytest -s tests/attention/test_trtllm_gen_attention.py -k "[True-" "$@"
