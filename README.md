# blasst-ae-mlsys26
Artifact Evaluation for MLSYS'26

## Reproducing Skip Softmax Attention Performance (Hopper FMHA Kernels)

This guide provides steps to reproduce the kernel-level performance benchmarking for the Skip Softmax Attention paper as described in the [TensorRT-LLM Tech Blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.html). It focuses on measuring attention sparsity and throughput across various threshold scale factors.

### 1. Launch the Container Environment
Start the interactive TensorRT-LLM container utilizing either Docker or Singularity (automatically determined based on permissions).

```bash
./start_docker.sh
```

### 2. Run the Benchmarks
Inside the container, run the provided benchmarking script (`benchmark_hopper.py`) to automatically sweep through the target sparsity levels. The script operates in two passes: first compiling the kernels with statistics enabled to measure actual sparsity, then recompiling without statistics to measure accurate performance throughput.

**Prefill Phase (Batch Size = 1):**
```bash
python3 benchmark_hopper.py --mode prefill
```

**Decode Phase (Batch Size = 1):**
```bash
python3 benchmark_hopper.py --mode decode
```

The script will output a tabular summary of the threshold, measured sparsity percentage, fused time execution (in microseconds), and the overall speedup compared to the baseline dense attention kernel.
