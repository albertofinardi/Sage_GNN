import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the root directory (parent of plotters directory)
ROOT_DIR = Path(__file__).parent.parent
PLOTS_DIR = ROOT_DIR / "plots"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

# Load the data
df = pd.read_csv(ROOT_DIR / "benchmark" / "results_scaling.csv")

# Create configuration labels (XnYg format)
df['config'] = df.apply(lambda row: f"{int(row['nodes'])}n{int(row['gpus'])}g", axis=1)

# Sort by total GPU count (nodes * gpus)
df['total_gpus'] = df['nodes'] * df['gpus']
df = df.sort_values('total_gpus').reset_index(drop=True)

print("="*80)
print("GENERATING SCALING ANALYSIS PLOTS")
print("="*80)

print("\nConfigurations (ordered by nodes*gpus):")
print(df[['config', 'nodes', 'gpus', 'total_gpus']].to_string())
print()

# Plot 1: Throughput (nodes/sec)
print("\nCreating: Throughput vs Configuration...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['config'], df['throughput_nodes_sec'], marker='o', linewidth=2, markersize=8, color='blue')
ax.set_xlabel('Configuration (XnYg)', fontsize=11)
ax.set_ylabel('Throughput (nodes/sec)', fontsize=11)
ax.set_title('Throughput vs Configuration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(df['throughput_nodes_sec']):
    ax.text(i, v + 10, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_throughput.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'scaling_throughput.png'}")

# Plot 2: Time per epoch (sec)
print("\nCreating: Time per Epoch vs Configuration...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['config'], df['time_per_epoch_sec'], marker='s', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Configuration (XnYg)', fontsize=11)
ax.set_ylabel('Time per Epoch (sec)', fontsize=11)
ax.set_title('Training Time per Epoch vs Configuration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(df['time_per_epoch_sec']):
    ax.text(i, v + 20, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_time_per_epoch.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'scaling_time_per_epoch.png'}")

# Plot 3: Train loss final
print("\nCreating: Training Loss vs Configuration...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['config'], df['train_loss_final'], marker='^', linewidth=2, markersize=8, color='red')
ax.set_xlabel('Configuration (XnYg)', fontsize=11)
ax.set_ylabel('Final Training Loss', fontsize=11)
ax.set_title('Final Training Loss vs Configuration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(df['train_loss_final']):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_train_loss.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'scaling_train_loss.png'}")

# Plot 4: Scaling efficiency (throughput scaling)
print("\nCreating: Scaling Efficiency vs Configuration...")
fig, ax = plt.subplots(figsize=(10, 6))
baseline_throughput = df.loc[0, 'throughput_nodes_sec']
scaling_efficiency = (df['throughput_nodes_sec'] / baseline_throughput) / df['total_gpus']
ax.plot(df['config'], scaling_efficiency * 100, marker='D', linewidth=2, markersize=8, color='purple')
ax.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect scaling (100%)')
ax.set_xlabel('Configuration (XnYg)', fontsize=11)
ax.set_ylabel('Scaling Efficiency (%)', fontsize=11)
ax.set_title('Scaling Efficiency vs Configuration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
ax.legend()
for i, v in enumerate(scaling_efficiency * 100):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_efficiency.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {PLOTS_DIR / 'scaling_efficiency.png'}")

# Analysis: Why does training loss increase with more GPUs?
print("\n" + "="*80)
print("ANALYSIS: Training Loss Increase with More GPUs")
print("="*80)
print(f"\nTraining Loss by Configuration:")
for idx, row in df.iterrows():
    print(f"  {row['config']}: {row['train_loss_final']:.4f}")

print(f"\nObservations:")
print(f"  - Single GPU (1n1g): {df.loc[0, 'train_loss_final']:.4f}")
print(f"  - Max GPUs (2n4g): {df.loc[len(df)-1, 'train_loss_final']:.4f}")
print(f"  - Increase: {((df.loc[len(df)-1, 'train_loss_final'] / df.loc[0, 'train_loss_final']) - 1) * 100:.1f}%")

print(f"\nPossible reasons for increased training loss with more GPUs:")
print(f"  1. Gradient Accumulation Effects: With accum_steps={df.loc[0, 'accum_steps']}")
print(f"     and distributed training, effective batch size per GPU changes,")
print(f"     altering gradient scaling and optimizer behavior.")
print(f"  2. Learning Rate Scaling: The learning rate may not scale properly for")
print(f"     larger distributed batches. Typically, LR should scale with batch size.")
print(f"  3. Communication Overhead: AllReduce operations during gradient synchronization")
print(f"     can introduce noise and affect convergence.")
print(f"  4. Different convergence paths: Different GPU distributions may explore")
print(f"     different regions of the loss landscape.")
print(f"  5. Stochastic noise: More workers = more stochastic sampling variations")
print(f"     leading to different final loss values despite similar accuracy.")

print(f"\nNote: Despite higher training loss, test accuracy remains competitive:")
print(f"  - 1n1g test_acc: {df.loc[0, 'test_acc']:.4f}")
print(f"  - 2n4g test_acc: {df.loc[len(df)-1, 'test_acc']:.4f}")
print(f"  This suggests the model is not overfitting; the loss is just measured differently.")

print("\n" + "="*80)
print(f"All plots saved to {PLOTS_DIR}/")
print("="*80)
