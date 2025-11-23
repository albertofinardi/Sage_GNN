#!/bin/sh -l
#SBATCH -J gnn                        # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (1 per GPU)
#SBATCH --output=gnn-%j.out           # Output file with job ID
#SBATCH --error=gnn-%j.err            # Error file with job ID
#SBATCH -c 7                          # Cores per task
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=4:00:00                # Time limit
#SBATCH -p gpu                        # Partition
#SBATCH -A p200981                    # Specify your account/project
#SBATCH --qos=default

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "=========================================="

# SOLUTION: Load PyTorch with CUDA support first!
# Use the latest stable environment
module load env/staging/2023.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo ""
echo "========================================"
echo "Starting GraphSAGE Training"
echo "========================================"
echo ""

python -u graphsage_ogbn_products.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job Completed"
echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"

# Exit with the same code as the Python script
exit $EXIT_CODE
