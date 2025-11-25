#!/bin/sh -l
#SBATCH -J gnn_setup                    # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (1 per GPU)
#SBATCH --output=gnn_setup_%j.out  # Output file with job ID
#SBATCH --error=gnn_setup_%j.err   # Error file with job ID
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=4:00:00                # Time limit
#SBATCH -p gpu                        # Partition
#SBATCH -A p200981                    # Specify your account/project
#SBATCH --qos=default

module load Apptainer

set -e

# Determine project directory (parent of container/)
PROJECT_DIR="${PROJECT}"
mkdir -p ${PROJECT_DIR}/GNN
CONTAINER_NAME="${PROJECT_DIR}/GNN/pytorch-gnn.sif"
DEF_FILE="pytorch-gnn.def"

echo "=========================================="
echo "Building Apptainer container..."
echo "=========================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Container will be saved to: ${CONTAINER_NAME}"
echo ""

# Build container
apptainer build --fakeroot ${CONTAINER_NAME} ${DEF_FILE}

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Container: ${CONTAINER_NAME}"
echo "Size: $(du -h ${CONTAINER_NAME} | cut -f1)"
echo ""
echo "Test with:"
echo "  apptainer exec --nv ${CONTAINER_NAME} nvidia-smi"
echo "=========================================="
