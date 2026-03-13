# BLASST Artifact Evaluation for MLSys 2026

## Overview

This repository contains the Artifact Evaluation (AE) code for BLASST, submitted to MLSys 2026. It provides the framework to reproduce kernel-level performance benchmarks for Skip-Softmax Attention.

Key components of this artifact evaluation include:
- **Phase Evaluation**: Benchmarking for both **prefill** and **decode** attention phases using custom Flash Multi-Head Attention (FMHA) kernels.
- **Supported Architectures**: Scripts and instructions targeting NVIDIA Hopper (H200) and Blackwell (B200) GPUs.
- **Measured Metrics**: Automated sweeps across various threshold scale factors to measure and report exact attention sparsity percentages, execution times, bandwidth, and speedups over dense baseline kernels.
- **Framework Integration**: Includes custom testing and benchmarking scripts built on top of TensorRT-LLM and FlashInfer.

Please refer to the specific phase and architecture subdirectories for detailed reproduction steps.

## Prerequisites

* H200 and B200 GPUs to collect respective performance numbers.
* Docker or Singularity.

## Quickstart

1. Clone the repo with external submodules

```
git clone git@github.com:cameronshinn/blasst-ae-mlsys26.git --recursive
```

2. Start the Docker container. Will fall back to Singularity if Docker is not available. The repository should get mounted to `/workspace`.

> [!IMPORTANT]
> If you use a cloud provider that launches an instance directly from a Docker image, you can point it to [nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc6).
> Some cloud service providers will mount `/workspace` to a slow network drive which drastically slow down build times. Use the home directory `/root` in this case.

```
cd blasst-ae-mlsys26 && ./start_docker.sh
cd /workspace
```

3. The artifacts are split across separate folers for prefill/decode and Hopper/Blackwell. Navigate to a directory and follow the steps there to reproduce the corresponding results.
