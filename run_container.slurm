#!/bin/sh -l
#SBATCH -J gnn                        # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node
#SBATCH --output=gnn_%j.out           # Output file with job ID
#SBATCH --error=gnn_%j.err            # Error file with job ID
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=4:00:00                # Time limit
#SBATCH -p gpu                        # Partition
#SBATCH -A p200981                    # Account
#SBATCH --qos=default

module load Apptainer

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Use the directory where sbatch was run
HOST_SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CONTAINER="${PROJECT}/GNN/pytorch-gnn.sif"
DATA_DIR="${PROJECT}/GNN/data"  # Store data in project directory
SCRATCH_DIR="${SCRATCH}/gnn_${SLURM_JOB_ID}"

# Create directories
mkdir -p ${DATA_DIR}
mkdir -p ${SCRATCH_DIR}

echo "Submit directory: ${HOST_SCRIPT_DIR}"
echo "Container: ${CONTAINER}"
echo "Data directory: ${DATA_DIR}"
echo "Scratch: ${SCRATCH_DIR}"
echo ""

# Download dataset on host if not already present
if [ ! -d "${DATA_DIR}/products" ]; then
    echo "Downloading dataset"
    cd ${DATA_DIR}
    wget --no-check-certificate https://snap.stanford.edu/ogb/data/nodeproppred/products.zip
    unzip products.zip
    rm products.zip
    echo "Dataset downloaded and extracted to ${DATA_DIR}"
    echo ""
fi

# Run with bindings - bind the data directory
apptainer exec --nv \
    --bind ${HOST_SCRIPT_DIR}:/workspace \
    --bind ${DATA_DIR}:/data \
    --bind ${SCRATCH_DIR}:/scratch \
    ${CONTAINER} \
    python /workspace/train_graphsage.py \
        --data_dir /data \
        --epochs 100 \
        --batch_size 2048 \
        --hidden_dim 256 \
        --num_layers 3

EXIT_CODE=$?
echo ""
echo "Training finished with exit code: ${EXIT_CODE}"

# Cleanup
rm -rf ${SCRATCH_DIR}

exit ${EXIT_CODE}