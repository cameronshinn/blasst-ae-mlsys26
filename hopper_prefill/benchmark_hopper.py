import sys
import re
import argparse
import subprocess
import threading
import time
import itertools

PRINT_COMPILATION_OUTPUT = False


class AnimatedPrint:
    def __init__(self, msg, disabled=False):
        self.msg = msg
        msg_clean = msg.rstrip(' .')
        self.base_msg = msg_clean
        self.disabled = disabled
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        spinner = itertools.cycle(['', '.', '..', '...'])
        while not self.stop_event.is_set():
            sys.stdout.write(f'\r{self.base_msg}{next(spinner):<3}')
            sys.stdout.flush()
            time.sleep(0.4)

    def __enter__(self):
        if self.disabled:
            print(self.msg, flush=True)
        else:
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disabled:
            self.stop_event.set()
            self.thread.join()
            sys.stdout.write(f'\r{self.base_msg}... {"(failed)" if exc_type else "(done)"}\n')
            sys.stdout.flush()


def build_kernels(enable_stats=False):
    """Builds the fmha kernels using the build_hopper.sh script."""
    command = ["bash", "build_hopper.sh"]
    if enable_stats:
        command.append("--skip-softmax-stat")
    try:
        msg = f"Compiling kernels (stats={'enabled' if enable_stats else 'disabled'})"
        with AnimatedPrint(msg, disabled=PRINT_COMPILATION_OUTPUT):
            if PRINT_COMPILATION_OUTPUT:
                subprocess.run(command, check=True)
            else:
                subprocess.run(command, check=True, capture_output=True)
        print("Kernel build successful!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to build kernels. Exit code {e.returncode}.", file=sys.stderr)
        if not PRINT_COMPILATION_OUTPUT and e.stderr:
            print(f"Error Output:\n{e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr}", file=sys.stderr)
        sys.exit(1)

def extract_fused_time(text):
    """Extracts the floating point value for 'Fused time' from fmha kernel output."""
    match = re.search(r"Fused time\s*\.*\:\s*([\d\.]+)\s*us", text)
    if match:
        return float(match.group(1))
    return None

def extract_sparsity(text):
    """Extracts the floating point percentage for 'Skip-Softmax' from fmha kernel output."""
    match = re.search(r"Skip-Softmax \.\:\s*\d+\s*\/\s*\d+\s*\=\s*([\d\.]+)\%", text)
    if match:
        return float(match.group(1))
    return None

def run_benchmark_and_extract(dtype, head_dim, batch_size, q_heads, kv_heads, seq_len, repeats, warmup_repeats, skip_softmax_scale_factor=None, seq_len_q=None):
    """Executes the fmha benchmark command with specified parameters and extracts the fused time."""
    command = [
        "TensorRT-LLM/cpp/kernels/fmha_v2/bin/fmha.exe",
        dtype,
        "-d", str(head_dim),
        "-b", str(batch_size),
        "-h", str(q_heads),
        "-gqa", str(kv_heads),
        "-s", str(seq_len),
        "-runs", str(repeats),
        "-warm-up-runs", str(warmup_repeats),
        "-skip-checks"
    ]

    if seq_len_q is not None:
        command.extend(["-s-q", str(seq_len_q)])
        command.extend(["-paged-kv"])

    # Append skip-softmax arguments only if a factor is provided
    if skip_softmax_scale_factor is not None:
        command.extend(["-skip-softmax-threshold-scale-factor", str(skip_softmax_scale_factor)])

    try:
        # Execute the command and capture standard output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        content = result.stdout

        fused_time = extract_fused_time(content)
        sparsity = extract_sparsity(content)

        # Extract TFLOP/s keyword is actually "Tensor core .."
        match_bw = re.search(r"Tensor core \.\.\:\s*([\d\.]+)\s*Tflop", content)
        bw = float(match_bw.group(1)) if match_bw else 0.0

        if fused_time is not None:
            return fused_time, sparsity, bw
        else:
            print("Error: 'Fused time' not found in the output.", file=sys.stderr)
            print("Raw command output:\n" + content, file=sys.stderr)
            return None, None, None

    except subprocess.CalledProcessError as e:
        print(f"Error: The benchmark command failed with exit code {e.returncode}.", file=sys.stderr)
        if e.stderr:
            print(f"Error Output: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: The fmha.exe binary could not be found. Please run this script from the workspace root containing 'TensorRT-LLM/'.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hopper FMHA kernels with Skip-Softmax.")
    args = parser.parse_args()

    factors_to_test = [
        0,       # Baseline (0%)
        10000,   #  0.00% sparsity
        30000,   #  5.70% sparsity
        35000,   # 10.23% sparsity
        40000,   # 23.78% sparsity
        45000,   # 32.07% sparsity
        50000,   # 40.72% sparsity
        53000,   # 49.24% sparsity
        55000,   # 57.27% sparsity
        60000,   # 64.56% sparsity
        75000,   # 75.80% sparsity
        100000,  # 84.85% sparsity
        130000,  # 89.84% sparsity
        150000,  # 92.06% sparsity
        200000,  # 94.47% sparsity
        1000000  # 99.42% sparsity
    ]
    seq_len_q_arg = None
    batch_size_arg = 1

    final_results = {(factor, seq, dtype): {"time": None, "sparsity": None, "bw": None}
                     for factor in factors_to_test
                     for seq in [16384, 65536]
                     for dtype in ["-bf16"]}

    print("=== Pass 1: Measuring Attention Sparsity ===")
    build_kernels(enable_stats=True)
    for seq in [16384, 65536]:
        for dtype in ["-bf16"]:
            for factor in factors_to_test:
                if factor == 0:
                    final_results[(factor, seq, dtype)]["sparsity"] = 0.0
                    continue

                with AnimatedPrint(f"Extracting sparsity for {dtype} seq={seq} threshold: {factor}"):
                    _, sparsity, _ = run_benchmark_and_extract(
                        dtype=dtype,
                        head_dim=128,
                        batch_size=batch_size_arg,
                        q_heads=64,
                        kv_heads=4,
                        seq_len=seq,
                        repeats=1,
                        warmup_repeats=0,
                        skip_softmax_scale_factor=factor,
                        seq_len_q=seq_len_q_arg
                    )
                final_results[(factor, seq, dtype)]["sparsity"] = sparsity

    print("\n=== Pass 2: Measuring Performance ===")
    build_kernels(enable_stats=False)
    for seq in [16384, 65536]:
        for dtype in ["-bf16"]:
            for factor in factors_to_test:
                with AnimatedPrint(f"Extracting performance for {dtype} seq={seq} threshold: {factor}"):
                    time_us, _, bw = run_benchmark_and_extract(
                        dtype=dtype,
                        head_dim=128,
                        batch_size=batch_size_arg,
                        q_heads=64,
                        kv_heads=4,
                        seq_len=seq,
                        repeats=10,
                        warmup_repeats=3,
                        skip_softmax_scale_factor=factor,
                        seq_len_q=seq_len_q_arg
                    )
                if time_us is not None:
                     final_results[(factor, seq, dtype)]["time"] = time_us
                     final_results[(factor, seq, dtype)]["bw"] = bw

    print(f"\n{'#' * 120}")
    print("# skipSoftmaxAttention Hopper Performance Data")
    print(f"{'#' * 120}")

    for seq in [16384, 65536]:
        seq_len_k = seq // 1024

        print(f"\n{'=' * 120}")
        print(f"Prefill phase    BS={batch_size_arg}    dH=128        num q heads 64    num kv heads 4")
        print(f"seqlen = {seq_len_k}k")
        print(f"{'=' * 120}")

        bf16_hdr = f"{'BF16':^55}"
        print(f"{'':>15}{bf16_hdr}")

        col_hdr = f"{'threshold':>10} {'sparsity %':>11} {'time/ms':>9} {'TFLOP/s':>10} {'Speedup':>8}"
        print(f"{'':>3}{col_hdr}")
        print("-" * 120)

        def get_baseline_time(dtype):
            return final_results.get((0, seq, dtype), {}).get("time")

        bf16_base_ms = get_baseline_time("-bf16") / 1000.0 if get_baseline_time("-bf16") else None

        def fmt_row(threshold_str, sparsity, time_ms, base_ms, bw):
            if time_ms is None:
                return " " * 55
            speedup = base_ms / time_ms if base_ms and time_ms > 0 else 0
            return f"{threshold_str:>10} {sparsity:>10.2f}% {time_ms:>9.3f} {bw:>10.3f} {speedup:>8.3f}"

        # Baseline rows (no-skip kernel)
        bf16_base_bw = final_results[(0, seq, "-bf16")]["bw"]

        bf16_base_row = fmt_row("0(NoSkip)", 0.0, bf16_base_ms, bf16_base_ms, bf16_base_bw) if bf16_base_ms else " " * 55
        print(f"   {bf16_base_row}")

        for factor in factors_to_test:
            if factor == 0: continue

            bf16_metrics = final_results[(factor, seq, "-bf16")]

            bf16_time = bf16_metrics["time"] / 1000.0 if bf16_metrics["time"] else None

            bf16_col = fmt_row(f"{factor / 100000.0:.3f}", bf16_metrics["sparsity"] or 0.0, bf16_time, bf16_base_ms, bf16_metrics["bw"] or 0.0)

            print(f"   {bf16_col}")
