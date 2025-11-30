# GraphSAGE Hyperparameter Exploration

## Base Configuration

The baseline configuration (`id 0`) serves as the reference point for all experiments:

| Parameter | Value |
|-----------|-------|
| `data_dir` | `/data` |
| `epochs` | `100` |
| `batch_size` | `128` |
| `hidden_dim` | `256` |
| `num_layers` | `5` |
| `num_neighbors` | `15 10 10 10 10` |
| `accum_steps` | `5` |
| `lr` | `0.003` |

## Exploration Strategy

We perform a systematic one-at-a-time parameter exploration to study the independent effect of each hyperparameter. For each parameter of interest, we evaluate different values while keeping all other parameters fixed at their baseline values.

## Parameter Variations

### `hidden_dim` (Hidden Dimension)
- `128`
- `160`
- `192`
- `224`
- `256` (baseline)

### `num_layers` (Number of GNN Layers)
- `2`
- `3`
- `4`
- `5` (baseline and maximum due to memory constraints)

### `batch_size` (Batch Size)
- `64`
- `80`
- `96`
- `112`
- `128` (baseline)

### `lr` (Learning Rate)
- `0.0005`
- `0.001`
- `0.003` (baseline)
- `0.005`
- `0.01`

### `accum_steps` (Gradient Accumulation Steps)
- `1`
- `2`
- `4`
- `5` (baseline)
- `8`

### `num_neighbors` (Neighborhood Sampling Per Layer)
**Note:** The number of values must match `num_layers`. 

- `10 10 10 10 10` (uniform conservative)
- `15 10 10 10 10` (baseline)
- `15 12 10 10 10` (slightly aggressive early layers)
- `12 12 12 10 10` (uniform mid-range)
- `10 10 10 10 5` (conservative last layer)
- `15 10 8 5 5` (progressive reduction)
