import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Get the root directory (parent of plotters directory)
ROOT_DIR = Path(__file__).parent.parent
PLOTS_DIR = ROOT_DIR / "plots"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
results = pd.read_csv(ROOT_DIR / "benchmark" / "results_minibatch.csv")

# Filter to only accum_steps=5
results = results[results['accum_steps'] == 5].copy()

# Filter out OOM runs for most plots (but keep for OOM analysis)
results_valid = results[results['oom_flag'] == 0].copy()

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("="*80)
print("GENERATING BATCH SIZE ANALYSIS PLOTS")
print("="*80)

# Time per Epoch vs Batch Size
print("\nCreating: Time per Epoch vs Batch Size...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results_valid['batch_size'], results_valid['time_per_epoch_sec'], 
        marker='o', linewidth=2.5, markersize=10, color='steelblue')
ax.set_xlabel("Batch Size", fontsize=13)
ax.set_ylabel("Time per Epoch (seconds)", fontsize=13)
ax.set_title("Time per Epoch vs Batch Size", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "batch_time_per_epoch_vs_batch_size.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'batch_time_per_epoch_vs_batch_size.png'}")

# Peak GPU Memory vs Batch Size
print("\nCreating: Peak GPU Memory vs Batch Size...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(top=37500)
ax.plot(results_valid['batch_size'], results_valid['peak_gpu_memory_mb'], 
        marker='o', linewidth=2.5, markersize=10, color='steelblue', label='Memory Usage')
# Add OOM limit line at 32.5 GB (32500 MB)
ax.axhline(y=32500, color='red', linestyle='--', linewidth=2.5, label='OOM Limit (32.5 GB)')
# Set xlim before fill_between to prevent it from extending
x_min, x_max = ax.get_xlim()
ax.set_xlim(x_min, x_max)
ax.fill_between([x_min, x_max], 32500, 37500, alpha=0.15, color='red')
# Add OOM text label centered in the red zone
ax.text((x_min + x_max) / 2, 35000, 'OOM', fontsize=16, fontweight='bold', color='red', 
        ha='center', va='center')
ax.set_xlabel("Batch Size", fontsize=13)
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13)
ax.set_title("Peak GPU Memory Usage vs Batch Size", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "batch_peak_memory_vs_batch_size.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'batch_peak_memory_vs_batch_size.png'}")

# Plot 3: Throughput vs Batch Size
print("\nCreating: Throughput vs Batch Size...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results_valid['batch_size'], results_valid['throughput_nodes_sec'], 
        marker='o', linewidth=2.5, markersize=10, color='steelblue')
ax.set_xlabel("Batch Size", fontsize=13)
ax.set_ylabel("Throughput (nodes/second)", fontsize=13)
ax.set_title("Training Throughput vs Batch Size", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "batch_throughput_vs_batch_size.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'batch_throughput_vs_batch_size.png'}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total runs: {len(results)}")
print(f"  Successful runs: {len(results_valid)}")
print(f"  OOM failures: {results['oom_flag'].sum()}")

print(f"\nBatch Size Range:")
print(f"  Min batch size: {results['batch_size'].min()}")
print(f"  Max batch size (successful): {results_valid['batch_size'].max()}")
print(f"  Max batch size (attempted): {results['batch_size'].max()}")

print(f"\nBest Configuration (by test accuracy):")
results_with_test = results_valid[results_valid['test_acc'] > 0]
if len(results_with_test) > 0:
    best_idx = results_with_test['test_acc'].idxmax()
    best_config = results_with_test.loc[best_idx]
    print(f"  Config ID: {best_config['id']}")
    print(f"  Test Accuracy: {best_config['test_acc']:.4f}")
    print(f"  Batch Size: {int(best_config['batch_size'])}")
    print(f"  Accum Steps: {int(best_config['accum_steps'])}")
    print(f"  Time per Epoch: {best_config['time_per_epoch_sec']:.2f}s")
    print(f"  Throughput: {best_config['throughput_nodes_sec']:.2f} nodes/s")
    print(f"  Peak GPU Memory: {best_config['peak_gpu_memory_mb']:.1f} MB")

print(f"\nPerformance Metrics (successful runs):")
print(f"  Avg Time per Epoch: {results_valid['time_per_epoch_sec'].mean():.2f}s (±{results_valid['time_per_epoch_sec'].std():.2f})")
print(f"  Avg Throughput: {results_valid['throughput_nodes_sec'].mean():.2f} nodes/s (±{results_valid['throughput_nodes_sec'].std():.2f})")
print(f"  Avg Peak Memory: {results_valid['peak_gpu_memory_mb'].mean():.1f} MB (±{results_valid['peak_gpu_memory_mb'].std():.1f})")

print("\n" + "="*80)
print(f"All plots saved to {PLOTS_DIR}/")
print("="*80)
