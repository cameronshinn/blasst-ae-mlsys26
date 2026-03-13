import argparse
import os
import sys
import pathlib
import csv
import pandas as pd

# Parse run-mode argument before importing flashinfer
parser = argparse.ArgumentParser(description="Benchmark Blackwell Skip Stats")
parser.add_argument("--run-mode", type=str, default="stats", choices=["stats", "perf"], help="Choose run purpose: stats or perf")
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
parser.add_argument("--print-results", action="store_true", default=False, help="Print results to stdout")
args, unknown = parser.parse_known_args()

# Set environment variables before importing flashinfer
os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(os.path.dirname(__file__), "flashinfer_stats" if args.run_mode == "stats" else "flashinfer")
os.environ["FLASHINFER_CUBIN_CHECKSUM_DISABLED"] = "1"

import torch
import numpy as np
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
    in_kv_lens = torch.full((batch_size,), max_in_kv_len, dtype=torch.int32, device=GPU_DEVICE)
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

def bench_trtllm_prefill(
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


    if args.run_mode == "stats":
        # Collect stats
        # Ensure buffer is initialized to 0
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
        ms = None
        tflops = None
    else:  # perf
        # Warmup
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            q, kv_cache, workspace_buffer, page_table, seq_lens.to(GPU_DEVICE),
            torch.max(q_lens).item(), torch.max(seq_lens).item(),
            bmm1_scale, bmm2_scale, batch_size, q_indptr, kv_indptr, window_left,
            kv_layout=kv_layout,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

        # Measure time
        measurements = bench_gpu_time(
            lambda: flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                q, kv_cache, workspace_buffer, page_table, seq_lens.to(GPU_DEVICE),
                torch.max(q_lens).item(), torch.max(seq_lens).item(),
                bmm1_scale, bmm2_scale, batch_size, q_indptr, kv_indptr, window_left,
                kv_layout=kv_layout,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            ),
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )
        sparsity = None
        ms = np.median(measurements)
        # TFLOPS
        tflops = attention_tflops_per_sec_with_actual_seq_lens(
            q_lens, seq_lens, head_dim, head_dim, num_qo_heads, True, ms
        )

    return ms, tflops, sparsity

def bench_trtllm_decode(
    batch_size,
    seq_len,
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
    q_lens = torch.full((batch_size,), 1, dtype=torch.int32, device=GPU_DEVICE)
    in_kv_lens = torch.full((batch_size,), seq_len - 1, dtype=torch.int32, device=GPU_DEVICE)
    seq_lens = q_lens + in_kv_lens

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
    if args.run_mode == "stats":
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace_buffer,
            page_table,
            seq_lens.to(GPU_DEVICE),
            torch.max(seq_lens).item(),
            bmm1_scale,
            bmm2_scale,
            window_left,
            kv_layout=kv_layout,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            skip_softmax_stats_buffer=skip_softmax_stats_buffer,
        )
    else:
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace_buffer,
            page_table,
            seq_lens.to(GPU_DEVICE),
            torch.max(seq_lens).item(),
            bmm1_scale,
            bmm2_scale,
            window_left,
            kv_layout=kv_layout,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

    # Measure time
    measurements = bench_gpu_time(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace_buffer,
            page_table,
            seq_lens.to(GPU_DEVICE),
            torch.max(seq_lens).item(),
            bmm1_scale,
            bmm2_scale,
            window_left,
            kv_layout=kv_layout,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        ),
        dry_run_iters=warmup_iters,
        repeat_iters=repeat_iters,
    )
    ms = np.median(measurements)

    # Collect stats
    if args.run_mode == "stats":
        skip_softmax_stats_buffer.zero_()
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace_buffer,
            page_table,
            seq_lens.to(GPU_DEVICE),
            torch.max(seq_lens).item(),
            bmm1_scale,
            bmm2_scale,
            window_left,
            kv_layout=kv_layout,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            skip_softmax_stats_buffer=skip_softmax_stats_buffer,
        )
        stats = skip_softmax_stats_buffer.cpu().numpy()
        sparsity = stats[2] / stats[3] if stats[3] > 0 else 0
    else:
        sparsity = None

    # TFLOPS
    tflops = attention_tflops_per_sec_with_actual_seq_lens(
        q_lens, seq_lens, head_dim, head_dim, num_qo_heads, True, ms
    )

    return ms, tflops, sparsity


if __name__ == "__main__":
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 10:
        print("Blackwell skip stats benchmark requires Blackwell GPU (SM100+). skipping...")
        exit(0)

    # Results collection
    results = []

    thresholds = [0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 100.0]
    seq_lens_to_bench = [16384, 65536]


    decode_batch_size = 64
    prefill_batch_size = 1
    num_kv_heads = 4
    head_grp_size = 16
    num_qo_heads = num_kv_heads * head_grp_size
    head_dim = 128
    page_size = 16
    q_dtype = args.q_dtype
    kv_dtype = args.kv_dtype
    kv_layout = args.kv_layout
    window_left = args.window_left

    # Benchmark configs
    configs = [
        ("prefill", prefill_batch_size, 16384),
        ("prefill", prefill_batch_size, 65536),
        # ("decode", decode_batch_size, 16384),
        # ("decode", decode_batch_size, 65536),
    ]

    for mode, batch_size, seq_len in configs:
        if args.print_results:
            print("\n" + "="*100)
            print(f"{mode.capitalize()} phase    BS={batch_size}    dH={head_dim}    num_q_heads={num_qo_heads}    num_kv_heads={num_kv_heads}")
            print(f"seqlen = {seq_len}    dtype = {q_dtype}    layout = {kv_layout}")
            print("="*100)
            print(f"{'Threshold':>10} | {'Sparsity':>12} | {'Time (ms)':>10} | {'TFLOPS':>10} | {'Baseline':>10} | {'Speedup':>10}")
            print("-" * 105)
        baseline_ms = None
        if args.run_mode != "stats":
            # Collect true baseline (skips_softmax=False)
            if mode == "prefill":
                baseline_ms, _, _ = bench_trtllm_prefill(
                    batch_size, seq_len, 0, num_kv_heads, head_grp_size,
                    head_dim, page_size, q_dtype, kv_dtype, kv_layout, window_left,
                    skips_softmax=False,
                    warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters
                )
            # elif mode == "decode":
            #     baseline_ms, _, _ = bench_trtllm_decode(
            #         batch_size, seq_len, num_kv_heads, head_grp_size,
            #         head_dim, page_size, q_dtype, kv_dtype, kv_layout, window_left,
            #         skips_softmax=False,
            #         warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters
            #     )

        baseline_ms_str = f"{baseline_ms:10.3f}".strip() if baseline_ms is not None else ""

        for threshold in thresholds:
            actual_threshold = max(threshold * seq_len, 1e-30)
            if mode == "prefill":
                ms, tflops, sparsity = bench_trtllm_prefill(
                    batch_size, seq_len, 0, num_kv_heads, head_grp_size,
                    head_dim, page_size, q_dtype, kv_dtype, kv_layout, window_left,
                    skips_softmax=True, skip_threshold=actual_threshold,
                    warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters
                )
            # elif mode == "decode":
            #     ms, tflops, sparsity = bench_trtllm_decode(
            #         batch_size, seq_len, num_kv_heads, head_grp_size,
            #         head_dim, page_size, q_dtype, kv_dtype, kv_layout, window_left,
            #         skips_softmax=True, skip_threshold=actual_threshold,
            #         warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters
            #     )
            sparsity_str = f"{sparsity*100:10.2f}%".strip() if sparsity is not None else ""
            ms_str = f"{ms:10.3f}".strip() if ms is not None else ""
            tflops_str = f"{tflops:10.2f}".strip() if tflops is not None else ""
            if baseline_ms and ms:
                speedup = baseline_ms / ms
                speedup_str = f"{speedup:10.3f}".strip()
            else:
                speedup_str = ""

            if args.print_results:
                print(f"{threshold:>10.3f} | {sparsity_str:>12} | {ms_str:>10} | {tflops_str:>10} | {baseline_ms_str:>10} | {speedup_str:>10}")
            results.append((mode, batch_size, seq_len, threshold, sparsity_str, ms_str, tflops_str, baseline_ms_str, speedup_str))

    # Write results to CSV
    csv_path = f"{'stats_results.csv' if args.run_mode == 'stats' else 'perf_results.csv'}"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mode", "BatchSize", "SeqLen", "Threshold", "Sparsity", "Time (ms)", "TFLOPS", "Baseline Time (ms)", "Speedup"])
        for row in results:
            writer.writerow(row)

    print(f"Results written to {csv_path}")
