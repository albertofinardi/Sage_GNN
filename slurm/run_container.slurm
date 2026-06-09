#!/bin/sh -l
#SBATCH -J gnn                        # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node
#SBATCH --output=logs/gnn_%j.out           # Output file with job ID
#SBATCH --error=logs/gnn_%j.err            # Error file with job ID
#SBATCH --gres=gpu:1                  # GPUs per node
#SBATCH --time=00:20:00                # Time limit
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
CONTAINER="${CONTAINER:-${SCRATCH}/pytorch-gnn-${USER}.sif}"
# Store data in scratch (user-writable); persists across jobs
DATA_DIR="${SCRATCH}/gnn_data"
SCRATCH_DIR="${SCRATCH}/gnn_${SLURM_JOB_ID}"

# Create directories
mkdir -p ${DATA_DIR}
mkdir -p ${SCRATCH_DIR}

echo "Submit directory: ${HOST_SCRIPT_DIR}"
echo "Container: ${CONTAINER}"
echo "Data directory: ${DATA_DIR}"
echo "Scratch: ${SCRATCH_DIR}"
echo ""

# Download dataset on host if not already present (or corrupted)
OGB_DATASET_DIR="${DATA_DIR}/ogbn_products"
if [ ! -d "${OGB_DATASET_DIR}/processed" ]; then
    echo "Downloading ogbn-products dataset..."
    # Clean up any corrupted partial downloads
    rm -rf ${OGB_DATASET_DIR}
    rm -f ${DATA_DIR}/ogbn_products.zip
    
    cd ${DATA_DIR}
    wget -q --show-progress https://snap.stanford.edu/ogb/data/nodeproppred/products.zip -O ogbn_products.zip
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download dataset"
        exit 1
    fi
    
    echo "Extracting dataset..."
    unzip -q ogbn_products.zip
    rm ogbn_products.zip
    
    # Rename to match OGB expected folder name
    if [ -d "${DATA_DIR}/products" ] && [ ! -d "${OGB_DATASET_DIR}" ]; then
        mv ${DATA_DIR}/products ${OGB_DATASET_DIR}
    fi
    
    echo "Dataset ready at ${OGB_DATASET_DIR}"
fi

# Run with bindings - bind the data directory
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container file '${CONTAINER}' not found."
    echo "If you built the container to your scratch area, set the CONTAINER environment variable to its path before submitting, e.g.:"
    echo "  sbatch --export=CONTAINER=${SCRATCH}/pytorch-gnn-${USER}.sif run_container.slurm"
    exit 1
fi

# Run training
# Using smaller batch with gradient accumulation: 256 * 8 = 2048 effective batch
apptainer exec --nv \
    --bind ${HOST_SCRIPT_DIR}:/workspace \
    --bind ${DATA_DIR}:/data \
    --bind ${SCRATCH_DIR}:/scratch \
    ${CONTAINER} \
    python /workspace/train_graphsage.py \
        --data_dir /data \
        --epochs 100 \
        --batch_size 128 \
        --hidden_dim 256 \
        --num_layers 5 \
        --num_neighbors 15 10 10 10 10 \
        --num_workers 32 \
        --accum_steps 4 \
        --patience 10

EXIT_CODE=$?
echo ""
echo "Training finished with exit code: ${EXIT_CODE}"

# Cleanup
rm -rf ${SCRATCH_DIR}

exit ${EXIT_CODE}