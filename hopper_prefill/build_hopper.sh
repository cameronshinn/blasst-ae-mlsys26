#!/bin/bash

unset ENABLE_SM80
unset ENABLE_SM89
export ENABLE_SM90=1
export ENABLE_HMMA_FP32=1
unset ENABLE_SM120

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# pushd saves the current directory and changes to the new one
pushd "${SCRIPT_DIR}/TensorRT-LLM/cpp/kernels/fmha_v2/" > /dev/null || exit 1

export USE_CCACHE=1
# export CCACHE_DIR="/workspace/.ccache"
# export TMPDIR="/workspace/tmp"
if [[ "$1" == "--skip-softmax-stat" ]]; then
    export CXXFLAGS="-DSKIP_SOFTMAX_STAT"
    export CUDAFLAGS="-DSKIP_SOFTMAX_STAT"
else
    unset CXXFLAGS
    unset CUDAFLAGS
fi
mkdir -p "$TMPDIR"
make clean
python3 setup.py
make dirs
make -j bin/fmha.exe

# popd returns to the original directory
popd > /dev/null
