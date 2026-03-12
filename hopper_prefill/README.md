# Collecting Hopper Performance

## Reproducing Skip Softmax Attention Performance (Hopper FMHA Kernels)

This guide provides steps to reproduce the kernel-level performance benchmarking for the Skip Softmax Attention paper as described in the [TensorRT-LLM Tech Blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.html). It focuses on measuring attention sparsity and throughput across various threshold scale factors.

### 0. Initialize Submodules
Before proceeding, ensure you have initialized all git submodules to fetch the required TensorRT-LLM source code:

```bash
git submodule update --init --recursive
```

### 1. Launch the Container Environment
Start the interactive TensorRT-LLM container utilizing either Docker or Singularity (automatically determined based on permissions).

```bash
./../start_docker.sh
```

If you are using a cloud service that launches a container automatically from an image, you can pull from `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6`.

### 2. Permissions during Build (Optional)

> [!NOTE]
> If you run into permission issues while building the performance kernels regarding the `ccache` or system temp directory, you can go into `hopper_prefill/build_hopper.sh` and uncomment `CCACHE_DIR="/workspace/.ccache"` and `TMPDIR="/workspace/tmp"` to localize those directories to your mounted workspace.

### 3. Run the Benchmarks
Inside the container, run the provided benchmarking script (`benchmark_hopper.py`) to automatically sweep through the target sparsity levels. The script operates in two passes: first compiling the kernels with statistics enabled to measure actual sparsity, then recompiling without statistics to measure performance accurately.

**Prefill Phase:**
```bash
python3 benchmark_hopper.py --mode prefill
```

**Decode Phase:**
```bash
python3 benchmark_hopper.py --mode decode
```

The script will output a tabular summary of the threshold, measured sparsity percentage, fused time execution (in microseconds), and the overall speedup compared to the baseline dense attention kernel.
