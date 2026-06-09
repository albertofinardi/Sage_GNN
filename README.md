# GraphSAGE on OGBN-Products

Benchmarking [GraphSAGE](https://arxiv.org/abs/1706.02216) for node classification on the [OGBN-Products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) dataset, with a focus on scalable training on GPU clusters using MeluXina.

**Dataset:** 2.4M nodes, 61.1M edges, 47 features, 100 classes (Amazon product co-purchasing graph).

---

## Repository Structure

```
Sage_GNN/
├── train_graphsage.py          # Single-GPU training
├── train_graphsage_ddp.py      # Multi-GPU DDP training
│
├── configs/
│   └── hyperparams.csv         # Hyperparameter configurations for the sweep
│
├── slurm/
│   ├── run_container.slurm     # Single-GPU SLURM job
│   ├── run_container_ddp.slurm # Multi-GPU DDP SLURM job
│   └── hyperparams.slurm       # Array job for hyperparameter sweep
│
├── container/
│   ├── pytorch-gnn.def         # Apptainer container definition
│   ├── build_container_slurm.sh
│   └── test_container_slurm.sh
│
├── benchmark/
│   ├── hyperparams_minibatch.csv   # Batch size study configs
│   ├── hyperparams_neighbor.csv    # Neighbor sampling study configs
│   ├── hyperparams_scaling.csv     # DDP scaling study configs
│   ├── results_minibatch.csv
│   ├── results_neighbor.csv
│   └── results_scaling.csv
│
├── plotters/
│   ├── plot_batch.py           # Batch size analysis
│   ├── plot_neighbor.py        # Neighbor sampling analysis
│   └── plot_scaling.py         # DDP scaling analysis
│
├── plots/                      # Generated figures
├── models/                # Saved model weights
├── report/                     # LaTeX report + compiled PDF
└── presentation/               # Beamer slides + compiled PDF
```

---

## Setup

All experiments run inside an Apptainercontainer (PyTorch 2.1.2, CUDA 12.1, PyG).

**1. Build the container** (submit once; check if a `.sif` already exists first):

```bash
cd container
sbatch build_container_slurm.sh
```

**2. Test the container** (optional — verifies all dependencies):

```bash
sbatch test_container_slurm.sh
```

The dataset (`ogbn-products`) is downloaded automatically on first run and cached in `$SCRATCH/gnn_data`.

---

## Running Experiments

All `sbatch` commands should be run from the **repo root**.

### Single-GPU training

```bash
sbatch slurm/run_container.slurm
```

Default config: 5-layer GraphSAGE, hidden_dim=256, batch_size=128, 100 epochs, fanout=[15,10,10,10,10].

Set `CONTAINER` if your `.sif` is not in `$SCRATCH`:

```bash
sbatch --export=ALL,CONTAINER=/path/to/pytorch-gnn.sif slurm/run_container.slurm
```

### Multi-GPU DDP training (4 GPUs, 1 node)

```bash
sbatch slurm/run_container_ddp.slurm
```

### Hyperparameter sweep (SLURM array)

```bash
sbatch slurm/hyperparams.slurm
```

Reads configs from `configs/hyperparams.csv`. Each array task runs one DDP job across 4 GPUs.

---

## Generating Plots

After benchmark runs have produced results CSVs in `benchmark/`:

```bash
python plotters/plot_batch.py
python plotters/plot_neighbor.py
python plotters/plot_scaling.py
```

Figures are written to `plots/`.

---

## Key Results

Experiments were run on MeluXina (NVIDIA A100 GPUs) for 150 epochs unless stated otherwise.

| Study                 | Finding                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Batch size**        | batch_size=160 fastest training without going over memory limits on A100                                     |
| **Neighbor sampling** | Dropping the last-layer fanout (`[15,10,10,10,0]`) cuts epoch time by **2.7×** with no accuracy loss (81.1%) |
| **DDP scaling**       | Near-linear scaling: 1->4 GPUs achieves **3.7× speedup**; 8 GPUs (2 nodes) reaches **7.1×**                  |

Full analysis in [`report/graphsage_report.pdf`](report/graphsage_report.pdf).

---

## References

- Hamilton et al., [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NeurIPS 2017)
- Hu et al., [Open Graph Benchmark](https://arxiv.org/abs/2005.00687) (NeurIPS 2020)
