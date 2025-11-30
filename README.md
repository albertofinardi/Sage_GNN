# GraphSAGE

## @Tommaso

### Benchmarking Resources

* **docs/experiment_setup.md** – Overview of the current experiment setup
* **hyperparams.csv** – Input hyperparameter configurations
* **hyperparams.slurm** – Runs all configurations using SLURM arrays
* **train_graphsage_ddp.py** – Executes experiments according to the configuration
* **results.csv** – Stores the outcomes of the experiments
* **plot.py** – Script to visualize results

### TODO

1. Run experiments for 100 epochs (currently only 1)
2. Consider modifying or extending the setup, e.g. 
   * also test more computational parameters like `num_workers`, number of GPUs
   * run more configurations
   * improve visualization
3. Analyze and interpret the results

## How to Launch

1. **Build the container:**
Before running, check a container image (SIF) doesn't already exists in the project folder. If need to override, add the flag `--force` in the `apptainer build` command
   ```bash
   cd container
   sbatch build_container_slurm.sh
   ```

2. **Test the container (optional):**
Checks all dependencies inside the container are setup correctly.
   ```bash
   sbatch test_container_slurm.sh
   ```

3. **Run training:**
Starts the container binding the dataset (+ download if not available yet) and code, plus starts the `train_graphsage.py` script for training. 
   ```bash
   sbatch run_container.slurm
   ```
