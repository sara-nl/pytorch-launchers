#!/bin/bash
#SBATCH --job-name=torchrun_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Enable auto-detection of the backend
BACKEND=""

# Check if GPUs are available
if command -v nvidia-smi &> /dev/null; then
    BACKEND="nccl"
    echo "GPUs detected, using NCCL backend"
else
    BACKEND="gloo"
    echo "No GPUs detected, using Gloo backend"
fi

# No modules needed for this example

# Get SLURM variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
NUM_NODES=$SLURM_JOB_NUM_NODES
# Auto-detect number of GPUs per node if not set in environment
if [ -n "$SLURM_GPUS_PER_NODE" ]; then
    NUM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
else
    # Default to 4 GPUs per node if not specified
    NUM_GPUS_PER_NODE=4
    echo "Could not detect GPUs per node, defaulting to $NUM_GPUS_PER_NODE"
fi

# Print some information about the job
echo "Running on $(hostname)"
echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"

# Run torchrun with srun
srun torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  distributed_sum.py \
  --backend=$BACKEND
