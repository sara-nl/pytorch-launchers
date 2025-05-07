#!/bin/bash
#SBATCH --job-name=accelerate_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
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

# Auto-detect number of GPUs per node if not set in environment
if [ -n "$SLURM_GPUS_PER_NODE" ]; then
    NUM_PROCESSES_PER_NODE=$SLURM_GPUS_PER_NODE
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_PROCESSES_PER_NODE=$SLURM_GPUS_ON_NODE
else
    # Default to 4 GPUs per node if not specified
    NUM_PROCESSES_PER_NODE=4
    echo "Could not detect GPUs per node, defaulting to $NUM_PROCESSES_PER_NODE"
fi

# Get SLURM variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
NUM_NODES=$SLURM_JOB_NUM_NODES

# Print some information about the job
echo "Running on $(hostname)"
echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $NUM_NODES"
echo "Processes per node: $NUM_PROCESSES_PER_NODE"
echo "Node rank: $SLURM_NODEID"

# Run with accelerate
srun accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_PROCESSES_PER_NODE \
  --num_machines=$NUM_NODES \
  --machine_rank=$SLURM_NODEID \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  distributed_sum.py \
  --backend=$BACKEND
