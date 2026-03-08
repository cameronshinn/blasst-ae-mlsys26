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

def run_benchmark_and_extract(head_dim, batch_size, q_heads, kv_heads, seq_len, repeats, warmup_repeats, skip_softmax_scale_factor=None, seq_len_q=None):
    """Executes the fmha benchmark command with specified parameters and extracts the fused time."""
    command = [
        "TensorRT-LLM/cpp/kernels/fmha_v2/bin/fmha.exe",
        "-bf16",
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

        if fused_time is not None:
            return fused_time, sparsity
        else:
            print("Error: 'Fused time' not found in the output.", file=sys.stderr)
            print("Raw command output:\n" + content, file=sys.stderr)
            return None, None

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
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"], default="prefill",
                        help="Select prefill (default) or decode benchmark mode.")
    args = parser.parse_args()

    if args.mode == "prefill":
        factors_to_test = [
            None,    # Baseline (0%)
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
            # 120000,  # 88.45% sparsity
            130000,  # 89.84% sparsity
            150000,  # 92.06% sparsity
            200000,  # 94.47% sparsity
            1000000  # 99.42% sparsity
        ]
        seq_len_q_arg = None
        batch_size_arg = 1
    else:
        # Dummy scale factors for decode mode
        factors_to_test = [
            None,
            100,
            500,
            1000,
            5000,
        ]
        seq_len_q_arg = 1
        batch_size_arg = 1
    final_results = {factor: {"time": None, "sparsity": None} for factor in factors_to_test}

    print("=== Pass 1: Measuring Attention Sparsity ===")
    build_kernels(enable_stats=True)
    for factor in factors_to_test:
        if factor is None:
            print("Skipping sparsity run for baseline (None).", flush=True)
            final_results[factor]["sparsity"] = 0.0
            continue

        with AnimatedPrint(f"Extracting sparsity for skip_softmax_scale_factor: {factor}"):
            _, sparsity = run_benchmark_and_extract(
                head_dim=128,
                batch_size=batch_size_arg,
                q_heads=64,
                kv_heads=4,
                seq_len=65536,
                repeats=1,
                warmup_repeats=0,
                skip_softmax_scale_factor=factor,
                seq_len_q=seq_len_q_arg
            )
        final_results[factor]["sparsity"] = sparsity

    print("\n=== Pass 2: Measuring Performance ===")
    build_kernels(enable_stats=False)
    for factor in factors_to_test:
        with AnimatedPrint(f"Extracting performance for skip_softmax_scale_factor: {factor}"):
            time_us, _ = run_benchmark_and_extract(
                head_dim=128,
                batch_size=batch_size_arg,
                q_heads=64,
                kv_heads=4,
                seq_len=65536,
                repeats=10,
                warmup_repeats=3,
                skip_softmax_scale_factor=factor,
                seq_len_q=seq_len_q_arg
            )
        if time_us is not None:
             final_results[factor]["time"] = time_us

    print("\n=======================================================")
    print(f"{'Threshold':<15} | {'Sparsity':<10} | {'Time (us)':<12} | {'Speedup'}")
    print("=======================================================")

    baseline_time = final_results.get(None, {}).get("time")

    for factor in factors_to_test:
        metrics = final_results[factor]
        label = str(factor) if factor is not None else "None (Baseline)"

        sparsity_val = metrics['sparsity']
        time_val = metrics['time']

        sparsity_str = f"{sparsity_val:.2f}%" if sparsity_val is not None else "N/A"
        time_str = f"{time_val:.2f}" if time_val is not None else "N/A"

        speedup_str = "N/A"
        if time_val is not None and baseline_time is not None and time_val > 0:
            speedup = baseline_time / time_val
            speedup_str = f"{speedup:.3f}x"

        print(f"{label:<15} | {sparsity_str:<10} | {time_str:<12} | {speedup_str}")
