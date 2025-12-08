import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Get the root directory (parent of plotters directory)
ROOT_DIR = Path(__file__).parent.parent
PLOTS_DIR = ROOT_DIR / "plots"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
results = pd.read_csv(ROOT_DIR / "benchmark" / "results_neighbor.csv")

# Calculate Total Fanout Factor (product of all neighbor samples, excluding zeros)
def calculate_fanout(num_neighbors_str):
    """Calculate product of all neighbor sample sizes (excluding zeros)"""
    try:
        numbers = [int(x) for x in num_neighbors_str.split()]
        # Filter out zeros
        non_zero_numbers = [n for n in numbers if n != 0]
        if not non_zero_numbers:
            return 0
        product = 1
        for n in non_zero_numbers:
            product *= n
        return product
    except:
        return 0

results['total_fanout'] = results['num_neighbors'].apply(calculate_fanout)

# Filter out OOM runs and test_acc=0 for most plots
results_valid = results[(results['oom_flag'] == 0) & (results['test_acc'] > 0)].copy()

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("="*80)
print("GENERATING NEIGHBOR SAMPLING ANALYSIS PLOTS")
print("="*80)

# ============================================================================
# PLOT 1: Accuracy vs. Throughput (Pareto Frontier)
# ============================================================================
print("\nCreating: Accuracy vs. Throughput (Pareto Frontier)...")
fig, ax = plt.subplots(figsize=(12, 7))

# Create scatter plot with color representing total fanout
scatter = ax.scatter(results_valid['throughput_nodes_sec'], 
                     results_valid['test_acc'],
                     c=results_valid['total_fanout'],
                     s=150,
                     alpha=0.7,
                     cmap='viridis',
                     edgecolors='black',
                     linewidth=1.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Fanout Factor', fontsize=12, fontweight='bold')

ax.set_xlabel("Throughput (nodes/second)", fontsize=13, fontweight='bold')
ax.set_ylabel("Test Accuracy", fontsize=13, fontweight='bold')
ax.set_title("Accuracy vs. Throughput: Pareto Frontier\n(Color = Total Fanout Factor)", 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "neighbor_accuracy_vs_throughput.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'neighbor_accuracy_vs_throughput.png'}")


# ============================================================================
# PLOT 2: Top 5 Test Accuracies Table (LaTeX)
# ============================================================================
print("\nCreating: Top 5 Test Accuracies LaTeX Table...")

# Get top 5 by test accuracy
top5 = results_valid.nlargest(5, 'test_acc')[['id', 'num_neighbors', 'test_acc', 'training_time_s']].copy()
top5['training_time_hours'] = top5['training_time_s'] / 60 / 60

# Create LaTeX table
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Top 5 Configurations by Test Accuracy}
\label{tab:top5_accuracy}
\begin{tabular}{clcc}
\toprule
\textbf{Rank} & \textbf{Neighbor Sampling} & \textbf{Test Acc} & \textbf{Training Time (h)} \\
\midrule
"""

for i, (idx, row) in enumerate(top5.iterrows(), 1):
    neighbor_str = row['num_neighbors'].replace(' ', ', ')
    # Bold the baseline (id=0)
    if row['id'] == 0:
        latex_table += f"{i} & \\textbf{{{neighbor_str}}} & \\textbf{{{row['test_acc']:.4f}}} & \\textbf{{{row['training_time_hours']:.2f}}} \\\\\n"
    else:
        latex_table += f"{i} & {neighbor_str} & {row['test_acc']:.4f} & {row['training_time_hours']:.2f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save to file
with open(PLOTS_DIR / "neighbor_top5_accuracy_table.tex", "w") as f:
    f.write(latex_table)

print(f"    ✓ Saved: {PLOTS_DIR / 'neighbor_top5_accuracy_table.tex'}")

# ============================================================================
# PLOT 3: Memory Usage vs. Total Fanout Factor
# ============================================================================
print("\nCreating: Memory Usage vs. Total Fanout Factor...")
fig, ax = plt.subplots(figsize=(10, 6))

valid_fanout = results_valid['total_fanout'].values
valid_memory = results_valid['peak_gpu_memory_mb'].values

# Plot memory usage (valid runs)
ax.plot(results_valid['total_fanout'], results_valid['peak_gpu_memory_mb'], 
        marker='o', linewidth=0, markersize=10, color='steelblue', alpha=0.7, label='Successful runs')

# Add trend line using square root fit (always increasing, sublinear growth)
# Fit: memory = a * sqrt(fanout) + b
sqrt_fanout = np.sqrt(valid_fanout)
z = np.polyfit(sqrt_fanout, valid_memory, 1)
p = np.poly1d(z)
x_smooth = np.linspace(valid_fanout.min(), valid_fanout.max(), 100)
ax.plot(x_smooth, p(np.sqrt(x_smooth)), "--", color="steelblue", alpha=0.5, linewidth=2, label='Trend (sqrt)')

# Add OOM limit line at 32.5 GB
ax.axhline(y=32500, color='red', linestyle='--', linewidth=2.5, label='OOM Limit (32.5 GB)')
x_min, x_max = ax.get_xlim()
y_min, y_max = 0, 37500
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.fill_between([x_min, x_max], 32500, y_max, alpha=0.15, color='red')
ax.text((x_min + x_max) / 2, 35000, 'OOM', fontsize=16, fontweight='bold', color='red', 
        ha='center', va='center')

ax.set_xlabel("Total Fanout Factor", fontsize=13, fontweight='bold')
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13, fontweight='bold')
ax.set_title("Peak GPU Memory vs. Total Fanout Factor", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "neighbor_memory_vs_fanout.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'neighbor_memory_vs_fanout.png'}")

# ============================================================================
# PLOT 4: Training Efficiency (Time vs Accuracy)
# ============================================================================
print("\nCreating: Training Efficiency (Time per Epoch vs Accuracy)...")
fig, ax = plt.subplots(figsize=(10, 6))

# Create scatter plot
scatter = ax.scatter(results_valid['time_per_epoch_sec'], 
                     results_valid['test_acc'],
                     c=results_valid['total_fanout'],
                     s=150,
                     alpha=0.7,
                     cmap='plasma',
                     edgecolors='black',
                     linewidth=1.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Fanout Factor', fontsize=12, fontweight='bold')

ax.set_xlabel("Time per Epoch (seconds)", fontsize=13, fontweight='bold')
ax.set_ylabel("Test Accuracy", fontsize=13, fontweight='bold')
ax.set_title("Training Efficiency: Time vs. Accuracy", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "neighbor_efficiency_time_vs_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'neighbor_efficiency_time_vs_accuracy.png'}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total configurations tested: {len(results)}")
print(f"  Successful runs: {len(results_valid)}")
print(f"  OOM failures: {results['oom_flag'].sum()}")

print(f"\nTotal Fanout Factor Range:")
print(f"  Min fanout: {results_valid['total_fanout'].min()}")
print(f"  Max fanout (successful): {results_valid['total_fanout'].max()}")
print(f"  Max fanout (attempted): {results['total_fanout'].max()}")

print(f"\nBest Configuration (by test accuracy):")
results_with_test = results_valid[results_valid['test_acc'] > 0]
if len(results_with_test) > 0:
    best_idx = results_with_test['test_acc'].idxmax()
    best_config = results_with_test.loc[best_idx]
    print(f"  Config ID: {best_config['id']}")
    print(f"  Neighbor Sampling: {best_config['num_neighbors']}")
    print(f"  Total Fanout: {int(best_config['total_fanout'])}")
    print(f"  Test Accuracy: {best_config['test_acc']:.4f}")
    print(f"  Training Time: {best_config['training_time_s']/60:.2f} min")
    print(f"  Time per Epoch: {best_config['time_per_epoch_sec']:.2f}s")
    print(f"  Throughput: {best_config['throughput_nodes_sec']:.2f} nodes/s")
    print(f"  Peak GPU Memory: {best_config['peak_gpu_memory_mb']:.1f} MB")

print(f"\nPerformance Metrics (successful runs):")
print(f"  Avg Test Accuracy: {results_valid['test_acc'].mean():.4f} (±{results_valid['test_acc'].std():.4f})")
print(f"  Avg Time per Epoch: {results_valid['time_per_epoch_sec'].mean():.2f}s (±{results_valid['time_per_epoch_sec'].std():.2f})")
print(f"  Avg Throughput: {results_valid['throughput_nodes_sec'].mean():.2f} nodes/s (±{results_valid['throughput_nodes_sec'].std():.2f})")
print(f"  Avg Peak Memory: {results_valid['peak_gpu_memory_mb'].mean():.1f} MB (±{results_valid['peak_gpu_memory_mb'].std():.1f})")
print(f"  Avg Total Fanout: {results_valid['total_fanout'].mean():.1f} (±{results_valid['total_fanout'].std():.1f})")

print("\n" + "="*80)
print(f"All plots saved to {PLOTS_DIR}/")
print("="*80)
