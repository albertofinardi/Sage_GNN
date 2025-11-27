#!/bin/sh -l
#SBATCH -J gnn_setup                    # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (1 per GPU)
#SBATCH --output=../logs/gnn_setup_%j.out  # Output file with job ID
#SBATCH --error=../logs/gnn_setup_%j.err   # Error file with job ID
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=4:00:00                # Time limit
#SBATCH -p gpu                        # Partition
#SBATCH -A p200981                    # Specify your account/project
#SBATCH --qos=default

module load Apptainer

set -e

# Determine project directory
PROJECT_DIR="${PROJECT}"
mkdir -p ${PROJECT_DIR}/GNN || true

# Prefer writing the container to a user-writable location to avoid permission
# errors when the project area is owned by a different account. Use $SCRATCH
# (fast, per-user) by default; override by setting CONTAINER env var.
CONTAINER_NAME="${CONTAINER:-${SCRATCH}/pytorch-gnn-${USER}.sif}"
echo "Using container output: ${CONTAINER_NAME}"
DEF_FILE="pytorch-gnn.def"

# Use scratch for build cache (faster I/O)
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH}/apptainer_tmp"
mkdir -p ${APPTAINER_CACHEDIR} ${APPTAINER_TMPDIR}

echo "=========================================="
echo "Building Apptainer container..."
echo "=========================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Container will be saved to: ${CONTAINER_NAME}"
echo "Cache directory: ${APPTAINER_CACHEDIR}"
echo "Temp directory: ${APPTAINER_TMPDIR}"
echo ""

# Build container (--force overwrites existing container without needing delete permissions)
apptainer build --fakeroot --force ${CONTAINER_NAME} ${DEF_FILE}

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Container: ${CONTAINER_NAME}"
echo "Size: $(du -h ${CONTAINER_NAME} | cut -f1)"
echo ""
echo "Test with:"
echo "  apptainer exec --nv ${CONTAINER_NAME} python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Run training with:"
echo "  cd ${SLURM_SUBMIT_DIR}/.."
echo "  sbatch run_container.slurm"
echo "=========================================="

# Cleanup temp directories
rm -rf ${APPTAINER_TMPDIR}/*
