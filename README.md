# GraphSAGE

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
