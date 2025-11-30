# Modules needed on Meluxina
# module load geopandas/1.0.1-foss-2024a
# module load Seaborn/0.13.2-gfbf-2024a

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results

results = pd.read_csv("results.csv")

# Set plotting style

sns.set_style("whitegrid")
sns.set_palette("husl")

# Create figure with subplots for each hyperparameter

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("GraphSAGE Hyperparameter Exploration Results", fontsize=16, fontweight='bold')

# Plot 1: Hidden Dimension

ax = axes[0, 0]
sns.lineplot(data=results, x="hidden_dim", y="test_acc", marker="o", ax=ax, linewidth=2)
ax.set_xlabel("Hidden Dimension", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Hidden Dimension vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: Number of Layers

ax = axes[0, 1]
sns.lineplot(data=results, x="num_layers", y="test_acc", marker="o", ax=ax, linewidth=2)
ax.set_xlabel("Number of Layers", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Number of Layers vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Batch Size

ax = axes[0, 2]
sns.lineplot(data=results, x="batch_size", y="test_acc", marker="o", ax=ax, linewidth=2)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Batch Size vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Learning Rate (log scale)

ax = axes[1, 0]
sns.lineplot(data=results, x="lr", y="test_acc", marker="o", ax=ax, linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Learning Rate vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 5: Gradient Accumulation Steps

ax = axes[1, 1]
sns.lineplot(data=results, x="accum_steps", y="test_acc", marker="o", ax=ax, linewidth=2)
ax.set_xlabel("Gradient Accumulation Steps", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Accumulation Steps vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 6: Training Time

ax = axes[1, 2]
sns.scatterplot(data=results, x="training_time_s", y="test_acc", s=100, ax=ax, alpha=0.7)
ax.set_xlabel("Training Time (seconds)", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Training Time vs Accuracy", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hyperparameter_exploration.png", dpi=300, bbox_inches='tight')
print("Plot saved as hyperparameter_exploration.png")
plt.show()

# Print summary statistics

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nBest configuration (by test accuracy):")
best_idx = results['test_acc'].idxmax()
best_config = results.loc[best_idx]
print(f"  Config ID: {best_config['id']}")
print(f"  Test Accuracy: {best_config['test_acc']:.4f}")
print(f"  Hidden Dim: {best_config['hidden_dim']}")
print(f"  Num Layers: {best_config['num_layers']}")
print(f"  Batch Size: {best_config['batch_size']}")
print(f"  Learning Rate: {best_config['lr']}")
print(f"  Accum Steps: {best_config['accum_steps']}")
print(f"  Training Time: {best_config['training_time_s']:.1f}s")

print(f"\nOverall statistics:")
print(f"  Mean Test Accuracy: {results['test_acc'].mean():.4f}")
print(f"  Std Test Accuracy: {results['test_acc'].std():.4f}")
print("="*80)

