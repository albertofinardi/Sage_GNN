#!/bin/bash -l
#SBATCH --job-name=hyperparams
#SBATCH --account=p200981
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=none
#SBATCH --time=10:00:00
#SBATCH --array=0-30                        # Array indices (match CSV IDs)
#SBATCH --output=logs/hyperparams_%A_%a.out
#SBATCH --error=logs/hyperparams_%A_%a.err

module load Apptainer

# Distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

# Container and data
CONTAINER="/mnt/tier1/project/p200981/pytorch-gnn-u103056.sif"
DATA_DIR="${SCRATCH}/gnn_data"
WORKSPACE="${SLURM_SUBMIT_DIR}"
SCRATCH_DIR="${SCRATCH}/gnn_ddp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${SCRATCH_DIR}

# ============================================================================

# Read hyperparameters from CSV for this array task
CONFIG_CSV="${WORKSPACE}/hyperparams.csv"
CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" ${CONFIG_CSV}) # skip header
IFS=',' read -r id hidden_dim num_layers batch_size lr accum_steps num_neighbors_str <<< "${CONFIG_LINE}"
num_neighbors=(${num_neighbors_str})

echo "Running config ID $id: hidden_dim=${hidden_dim}, num_layers=${num_layers}, batch_size=${batch_size}, lr=${lr}, accum_steps=${accum_steps}, num_neighbors=${num_neighbors[@]}"

# ============================================================================

# Launch DDP training
srun apptainer exec --nv \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA_DIR}:/data \
    --bind ${SCRATCH_DIR}:/scratch \
    ${CONTAINER} \
    python /workspace/train_graphsage_ddp.py \
        --data_dir /data \
        --epochs 1 \
        --batch_size ${batch_size} \
        --hidden_dim ${hidden_dim} \
        --num_layers ${num_layers} \
        --num_neighbors ${num_neighbors[@]} \
        --num_workers 4 \
        --accum_steps ${accum_steps} \
        --patience 10 \
        --results_file /workspace/results.csv \
        --config_id ${id}

EXIT_CODE=$?
rm -rf ${SCRATCH_DIR}
exit ${EXIT_CODE}
