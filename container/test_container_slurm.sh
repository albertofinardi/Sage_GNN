#!/bin/sh -l
#SBATCH -J gnn_test                    # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (1 per GPU)
#SBATCH --output=gnn_test_%j.out  # Output file with job ID
#SBATCH --error=gnn_test_%j.err   # Error file with job ID
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=4:00:00                # Time limit
#SBATCH -p gpu                        # Partition
#SBATCH -A p200981                    # Specify your account/project
#SBATCH --qos=default

module load Apptainer

set -e

PROJECT_DIR="${PROJECT}"
CONTAINER="${PROJECT_DIR}/GNN/pytorch-gnn.sif"


if [ ! -f "${CONTAINER}" ]; then
    echo "Error: Container not found at ${CONTAINER}"
    echo "Please build first"
    exit 1
fi

echo "=========================================="
echo "Testing container: ${CONTAINER}"
echo "=========================================="

echo ""
echo "1. Testing GPU access..."
apptainer exec --nv ${CONTAINER} nvidia-smi

echo ""
echo "2. Testing PyTorch..."
apptainer exec --nv ${CONTAINER} python3 -c \
  "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "3. Testing PyTorch Geometric..."
apptainer exec --nv ${CONTAINER} python3 -c \
  "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

echo ""
echo "4. Testing NeighborLoader..."
apptainer exec --nv ${CONTAINER} python3 -c \
  "from torch_geometric.loader import NeighborLoader; print('NeighborLoader: OK')"

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
