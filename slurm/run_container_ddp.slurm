#!/bin/bash -l
#SBATCH --job-name=gnn_ddp
#SBATCH --account=p200981
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1                      # Number of nodes (start with 1)
#SBATCH --ntasks=4                     # Total number of processes (= total GPUs)
#SBATCH --ntasks-per-node=4            # Processes per node (= GPUs per node, max 4 on MeluXina)
#SBATCH --gpus-per-task=1              # 1 GPU per process
#SBATCH --cpus-per-task=8              # CPU cores per task for data loading
#SBATCH --gpu-bind=none                # REQUIRED for NCCL - don't bind GPUs to tasks
#SBATCH --time=02:00:00
#SBATCH --output=logs/gnn_ddp_%j.out
#SBATCH --error=logs/gnn_ddp_%j.err

# Load required modules
module load Apptainer

# ============================================================================
# DISTRIBUTED TRAINING ENVIRONMENT
# ============================================================================

# Set master address (first node in allocation)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Set master port (derive from job ID to avoid collisions)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# World size = total number of tasks
export WORLD_SIZE=$SLURM_NTASKS

echo "=========================================="
echo "DDP Training Configuration"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs (world size): $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# ============================================================================
# NCCL CONFIGURATION (optimized for MeluXina InfiniBand)
# ============================================================================

export NCCL_DEBUG=WARN                    # Set to INFO for debugging
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_IB_HCA=mlx5                   # Use Mellanox HCA
export NCCL_SOCKET_IFNAME=ib0             # Use InfiniBand interface

# ============================================================================
# PATHS
# ============================================================================

CONTAINER="${CONTAINER:-${SCRATCH}/pytorch-gnn-${USER}.sif}"
DATA_DIR="${SCRATCH}/gnn_data"
WORKSPACE="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/gnn_ddp_${SLURM_JOB_ID}"

# Create directories
mkdir -p ${SCRATCH_DIR}

echo ""
echo "Container: ${CONTAINER}"
echo "Data directory: ${DATA_DIR}"
echo "Workspace: ${WORKSPACE}"
echo ""

# ============================================================================
# DATASET DOWNLOAD (run once before distributed training starts)
# ============================================================================

OGB_DATASET_DIR="${DATA_DIR}/ogbn_products"
if [ ! -d "${OGB_DATASET_DIR}/processed" ]; then
    echo "Downloading ogbn-products dataset..."
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
    
    if [ -d "${DATA_DIR}/products" ] && [ ! -d "${OGB_DATASET_DIR}" ]; then
        mv ${DATA_DIR}/products ${OGB_DATASET_DIR}
    fi
    
    echo "Dataset ready at ${OGB_DATASET_DIR}"
fi

# ============================================================================
# CONTAINER CHECK
# ============================================================================

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container file '${CONTAINER}' not found."
    echo "Build the container first with:"
    echo "  cd ${WORKSPACE}/container && sbatch build_container_slurm.sh"
    exit 1
fi

# ============================================================================
# LAUNCH DISTRIBUTED TRAINING
# ============================================================================

echo ""
echo "Launching DDP training with srun..."
echo ""

# srun launches $SLURM_NTASKS copies of the command
# Each task gets SLURM_PROCID (global rank) and SLURM_LOCALID (local rank)
# Reduce batch size to avoid OOM; use accum_steps to maintain effective batch
# Effective batch = 128 * 4 accum * 4 GPUs = 2048
srun apptainer exec --nv \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA_DIR}:/data \
    --bind ${SCRATCH_DIR}:/scratch \
    ${CONTAINER} \
    python /workspace/train_graphsage_ddp.py \
        --data_dir /data \
        --epochs 100 \
        --batch_size 128 \
        --hidden_dim 256 \
        --num_layers 5 \
        --num_neighbors 15 10 10 10 10 \
        --num_workers 4 \
        --accum_steps 4 \
        --patience 10

EXIT_CODE=$?
echo ""
echo "Training finished with exit code: ${EXIT_CODE}"

# Cleanup
rm -rf ${SCRATCH_DIR}

exit ${EXIT_CODE}
