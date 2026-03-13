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
    merged = pd.merge(stats_df, perf_df, on=["Mode", "BatchSize", "SeqLen", "Threshold"], how="outer", suffixes=("_stats", "_perf"))
    # Fill missing values from each side
    for col in ["Time (ms)", "TFLOPS", "Sparsity", "Baseline Time (ms)", "Speedup"]:
        stats_col = f"{col}_stats"
        perf_col = f"{col}_perf"
        if stats_col in merged.columns and perf_col in merged.columns:
            merged[col] = merged[stats_col].combine_first(merged[perf_col])
        elif stats_col in merged.columns:
            merged[col] = merged[stats_col]
        elif perf_col in merged.columns:
            merged[col] = merged[perf_col]

    # Select columns for output
    output_cols = ["Mode", "BatchSize", "SeqLen", "Threshold", "Sparsity", "Time (ms)", "TFLOPS", "Baseline Time (ms)", "Speedup"]
    merged[output_cols].to_csv("merged_results.csv", index=False)
    print("Merged results written to merged_results.csv\n")
    # Print nicely formatted table
    # Problem size constants (matching run_kernels.py)
    HEAD_DIM = 128
    NUM_KV_HEADS = 4
    NUM_Q_HEADS = 64

    # Group by Mode, BatchSize, SeqLen
    groups = merged.groupby(["Mode", "BatchSize", "SeqLen"], sort=False)

    for (mode, bs, seqlen), group in groups:
        print("\n" + "="*100)
        print(f"{mode.capitalize()} phase    BS={bs}    dH={HEAD_DIM}    num_q_heads={NUM_Q_HEADS}    num_kv_heads={NUM_KV_HEADS}")
        print(f"seqlen = {seqlen}")
        print("="*100)
        table_cols = ["Threshold", "Sparsity", "Time (ms)", "TFLOPS", "Baseline Time (ms)", "Speedup"]
        header_fmt = "{:>10} | {:>12} | {:>10} | {:>10} | {:>18} | {:>10}"
        row_fmt = "{:>10.3f} | {:>12} | {:>10.3f} | {:>10.2f} | {:>18.3f} | {:>10.3f}"
        print(header_fmt.format(*table_cols))
        print("-" * 105)
        for row in group[table_cols].itertuples(index=False):
            # Sparsity is already a string with % from run_kernels.py, but handle if it's not
            sparsity_val = row.Sparsity
            if isinstance(sparsity_val, (float, int)):
                sparsity_val = f"{sparsity_val*100:.2f}%"

            print(row_fmt.format(
                row.Threshold,
                sparsity_val,
                row[2] if pd.notnull(row[2]) and row[2] != "" else 0.0,
                row[3] if pd.notnull(row[3]) and row[3] != "" else 0.0,
                row[4] if pd.notnull(row[4]) and row[4] != "" else 0.0,
                row[5] if pd.notnull(row[5]) and row[5] != "" else 0.0
            ))
else:
    print("One or both CSV files missing, skipping merge.")
