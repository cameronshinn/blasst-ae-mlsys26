import subprocess
import sys
import csv
import os
import pandas as pd

MODES = ["stats", "perf"]
CSV_FILES = {"stats": "stats_results.csv", "perf": "perf_results.csv"}

for mode in MODES:
    print(f"\n=== Running for mode: {mode} ===")
    # Run install_flashinfer.sh, suppress output unless error
    try:
        subprocess.run(["./install_flashinfer.sh", mode], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running install_flashinfer.sh {mode}:\n{e}")
        sys.exit(1)
    # Run run_kernels.py, print output always
    ret = subprocess.run([sys.executable, "run_kernels.py", "--run-mode", mode], check=True)

# Merge CSVs using pandas outer join
if os.path.exists(CSV_FILES["stats"]) and os.path.exists(CSV_FILES["perf"]):
    stats_df = pd.read_csv(CSV_FILES["stats"])
    perf_df = pd.read_csv(CSV_FILES["perf"])
    merged = pd.merge(stats_df, perf_df, on=["SeqLen", "Threshold"], how="outer", suffixes=("_stats", "_perf"))
    # Fill missing values from each side
    for col in ["Time (ms)", "TFLOPS", "Sparsity", "Baseline Time (ms)", "Speedup"]:
        stats_col = f"{col}_stats"
        perf_col = f"{col}_perf"
        merged[col] = merged[stats_col].combine_first(merged[perf_col])
    # Select columns for output
    output_cols = ["SeqLen", "Threshold", "Time (ms)", "TFLOPS", "Sparsity", "Baseline Time (ms)", "Speedup"]
    merged[output_cols].to_csv("merged_results.csv", index=False)
    print("Merged results written to merged_results.csv\n")
    # Print nicely formatted table
    print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<16} {:<10}".format(*output_cols))
    print("-" * 74)
    for row in merged[output_cols].itertuples(index=False):
        print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<16} {:<10}".format(*[str(x) if x is not None else "" for x in row]))
else:
    print("One or both CSV files missing, skipping merge.")
