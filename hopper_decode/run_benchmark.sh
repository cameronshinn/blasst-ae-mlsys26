#!/bin/bash
# Wrapper script to run skip_softmax_perf for hopper_decode
# Make sure nvcc is in the PATH
export PATH="/usr/local/cuda/bin:$PATH"

# Run the benchmark from the TensorRT-LLM submodule
cd TensorRT-LLM/cpp/kernels/xqa
python run_skip_softmax_perf.py "$@"

# Parse and print the benchmark results
python parse_skip_softmax_log.py --log_dir skip_softmax_logs
