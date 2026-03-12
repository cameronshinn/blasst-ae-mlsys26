"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import pathlib
import torch
import numpy as np

# Must set environment variables BEFORE importing flashinfer
# because cubin directory paths are resolved at module import time.
_script_dir = pathlib.Path(__file__).resolve().parent
os.environ["FLASHINFER_WORKSPACE_BASE"] = str(_script_dir / "flashinfer")
os.environ["FLASHINFER_CUBIN_CHECKSUM_DISABLED"] = "1"

import flashinfer
from flashinfer.utils import get_compute_capability
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

GPU_DEVICE = "cuda:0"
workspace_size = 256 * 1024 * 1024
global_workspace_buffer = None

def to_float8(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def generate_seq_lens_prefill(batch_size: int, q_len: int, max_in_kv_len: int):
    q_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=GPU_DEVICE)
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int, device=GPU_DEVICE)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens

def generate_cumsum_lens(lens: torch.Tensor):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )

def create_query_tensor(q_lens: torch.Tensor, num_qo_heads: int, head_dim: int, q_dtype: str):
    q = torch.randn(
        int(torch.sum(q_lens).item()),
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
        device=GPU_DEVICE,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
    else:
        q_scale = 1.0
    return q, q_scale

def create_kv_cache(batch_size: int, seq_lens: torch.Tensor, page_size: int, num_kv_heads: int, head_dim: int, kv_dtype: str, kv_layout: str = "HND"):
    max_seq_len = torch.max(seq_lens).item()
    num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    num_pages = num_pages_per_seq * batch_size

    init_dtype = torch.bfloat16 if kv_dtype == "fp8" else DTYPE_MAP[kv_dtype]

    if kv_layout == "HND":
        k_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, dtype=init_dtype, device=GPU_DEVICE)
        v_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, dtype=init_dtype, device=GPU_DEVICE)
    else:
        k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=init_dtype, device=GPU_DEVICE)
        v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=init_dtype, device=GPU_DEVICE)

    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
    else:
        k_scale = v_scale = 1.0

    kv_cache = torch.stack([k_cache, v_cache], dim=1)
    return kv_cache, k_scale, v_scale

def create_page_table(batch_size: int, seq_lens: torch.Tensor, page_size: int):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = int(torch.max(page_per_seq).item())
    total_pages_needed = int(torch.sum(page_per_seq).item())
    all_page_ids = torch.randperm(total_pages_needed, dtype=torch.int32, device=GPU_DEVICE)
    page_tables = torch.zeros((batch_size, max_num_pages_per_seq), dtype=torch.int32, device=GPU_DEVICE)
    page_id = 0
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i]
        page_tables[i, :num_pages_needed] = all_page_ids[page_id : page_id + num_pages_needed]
        page_id += num_pages_needed
    return page_tables, page_per_seq

def get_last_page_len(seq_lens, page_size):
    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    return kv_last_page_len

def create_workspace_buffer():
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(workspace_size, dtype=torch.int8, device=GPU_DEVICE)
    return global_workspace_buffer

def bench_trtllm_skip_stats(
    batch_size,
    max_q_len,
    max_kv_len,
    num_kv_heads,
    head_grp_size,
    head_dim,
    page_size,
    q_dtype,
    kv_dtype,
    kv_layout,
    window_left,
    skips_softmax,
    skip_threshold=1e-30,
    warmup_iters=10,
    repeat_iters=50,
):
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)

    q, q_scale = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    kv_cache, k_scale, v_scale = create_kv_cache(batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, kv_layout)
    page_table, page_per_seq = create_page_table(batch_size, seq_lens, page_size)
    kv_indptr = generate_cumsum_lens(page_per_seq)

    workspace_buffer = create_workspace_buffer()

    sm_scale = float(1.0 / (head_dim**0.5))
    bmm1_scale = q_scale * k_scale * sm_scale
    bmm2_scale = v_scale # assuming o_scale = 1.0

    skip_softmax_threshold_scale_factor = skip_threshold if skips_softmax else None
    skip_softmax_stats_buffer = torch.zeros(4, device=GPU_DEVICE, dtype=torch.int32)

    # Warmup
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q, kv_cache, workspace_buffer, page_table, seq_lens.to(GPU_DEVICE),
        torch.max(q_lens).item(), torch.max(seq_lens).item(),
        bmm1_scale, bmm2_scale, batch_size, q_indptr, kv_indptr, window_left,
        kv_layout=kv_layout,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        skip_softmax_stats_buffer=skip_softmax_stats_buffer,
    )

    # Measure time
    measurements = bench_gpu_time(
        lambda: flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            q, kv_cache, workspace_buffer, page_table, seq_lens.to(GPU_DEVICE),
            torch.max(q_lens).item(), torch.max(seq_lens).item(),
            bmm1_scale, bmm2_scale, batch_size, q_indptr, kv_indptr, window_left,
            kv_layout=kv_layout,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            skip_softmax_stats_buffer=skip_softmax_stats_buffer,
        ),
        dry_run_iters=warmup_iters,
        repeat_iters=repeat_iters,
    )
    ms = np.median(measurements)

    # Collect stats (from the last run in bench_gpu_time, though they should be cumulative or we could reset)
    # Actually skip_softmax_stats_buffer is NOT reset in the lambda, so it accumulates.
    # To get accurate stats for a single run, we should reset it.
    skip_softmax_stats_buffer.zero_()
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q, kv_cache, workspace_buffer, page_table, seq_lens.to(GPU_DEVICE),
        torch.max(q_lens).item(), torch.max(seq_lens).item(),
        bmm1_scale, bmm2_scale, batch_size, q_indptr, kv_indptr, window_left,
        kv_layout=kv_layout,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        skip_softmax_stats_buffer=skip_softmax_stats_buffer,
    )
    stats = skip_softmax_stats_buffer.cpu().numpy()
    sparsity = stats[2] / stats[3] if stats[3] > 0 else 0

    # TFLOPS
    tflops = attention_tflops_per_sec_with_actual_seq_lens(
        q_lens, seq_lens, head_dim, head_dim, num_qo_heads, True, ms
    )

    return ms, tflops, sparsity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Blackwell Skip Stats")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-q-len", type=int, default=1)
    parser.add_argument("--max-kv-len", type=int, default=2048)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-grp-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--q-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--kv-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--kv-layout", type=str, default="NHD", choices=["HND", "NHD"])
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--skip-threshold", type=float, default=1e-30)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=50)
    args = parser.parse_args()

    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 10:
        print("Blackwell skip stats benchmark requires Blackwell GPU (SM100+). skipping...")
        exit(0)

    # Results collection
    results = []

    thresholds = [0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]
    seq_lens_to_bench = [16384, 65536]

    print(f"Configuration: BS={args.batch_size}, QLen={args.max_q_len}, heads={args.num_kv_heads}x{args.head_grp_size}, dim={args.head_dim}, page={args.page_size}, layout={args.kv_layout}, dtype={args.q_dtype}")
    print(f"{'SeqLen':>8} | {'Threshold':>10} | {'Time (ms)':>10} | {'TFLOPS':>10} | {'Sparsity':>10}")
    print("-" * 58)

    for seq_len in seq_lens_to_bench:
        for threshold in thresholds:
            # Avoid 0 threshold as it might disable the feature in the library,
            # which leads to "missing kernel" since this build only has skip kernels.
            actual_threshold = max(threshold, 1e-30)
            ms, tflops, sparsity = bench_trtllm_skip_stats(
                args.batch_size, args.max_q_len, seq_len, args.num_kv_heads, args.head_grp_size,
                args.head_dim, args.page_size, args.q_dtype, args.kv_dtype, args.kv_layout, args.window_left,
                skips_softmax=True, skip_threshold=actual_threshold,
                warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters
            )
            print(f"{seq_len:>8} | {threshold:>10.3f} | {ms:>10.3f} | {tflops:>10.2f} | {sparsity:>10.4f}")
            results.append((seq_len, threshold, ms, tflops, sparsity))
