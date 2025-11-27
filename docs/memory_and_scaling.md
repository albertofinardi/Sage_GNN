# GNN Memory & Scaling: Why DeepSpeed Doesn't Help

## The Memory Bottleneck in GNNs

GraphSAGE has only ~2M parameters, yet runs OOM on a 40GB A100. Why?

**The bottleneck is not the model — it's the sampled subgraphs.**

| Component | Memory Usage |
|-----------|--------------|
| Model parameters | ~8 MB |
| Optimizer states | ~24 MB |
| **Sampled subgraph + activations** | **30-40 GB** |

With 5 layers and fanout `[15, 10, 10, 10, 10]`, each seed node can expand to 150,000 neighbors. Multiply by batch size, and you get massive intermediate tensors.

## Observed Utilization (A100 40GB)

```
Memory:   37GB / 40GB (91%)  - Well utilized
GPU-Util: 100%               - Always busy  
Power:    158W / 400W (40%)  - Memory-bound, not compute-bound
```

GNNs are inherently **memory-bandwidth limited** due to irregular sparse access patterns. The 40% power draw is normal — we're waiting on memory, not starving for compute.

## Why Not DeepSpeed?

DeepSpeed's ZeRO optimizer shards **model parameters and optimizer states** across GPUs:

| ZeRO Stage | What it shards |
|------------|----------------|
| Stage 1 | Optimizer states |
| Stage 2 | + Gradients |
| Stage 3 | + Parameters |

**Our problem:** Parameters + optimizer = ~32 MB. Activations = ~37 GB.

DeepSpeed solves the wrong problem. ZeRO cannot shard activation memory from neighbor sampling — that's per-GPU, per-batch data.

## What Actually Helps

| Technique | Impact | Why |
|-----------|--------|-----|
| Reduce fanout | ★★★★★ | Exponentially reduces subgraph size |
| Smaller batch + gradient accumulation | ★★★★ | Smaller subgraphs, same effective batch |
| Fewer layers | ★★★★ | Fewer sampling hops |
| Multi-GPU (DDP) | ★★★ | Each GPU handles smaller batch |
| Mixed precision (FP16) | ★★ | Halves activation memory |
| Gradient checkpointing | ★★★ | Recompute vs store activations |

## When Multi-GPU Helps

Data parallelism (DDP) **does** help because:
- Each GPU processes a **smaller batch** independently
- Smaller batch → smaller sampled subgraph → fits in memory
- Gradients are synchronized, not activations

Example: Instead of 1 GPU × 512 batch, use 4 GPUs × 128 batch each.

## Current Configuration

```
Layers:      5
Fanout:      [15, 10, 10, 10, 10]
Batch size:  256
Accum steps: 8
Effective:   2048
```

This fits in 40GB by using small micro-batches with gradient accumulation — achieving large effective batch without large memory footprint.

## Summary

| Approach | Use for GNNs? |
|----------|---------------|
| DeepSpeed / ZeRO |  No — wrong bottleneck |
| Multi-GPU DDP |  Yes — splits batch |
| Gradient accumulation | Yes — our current approach |
| Reduce fanout/layers |  Yes — most effective |
