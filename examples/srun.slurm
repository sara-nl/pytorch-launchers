#!/bin/bash
#SBATCH --job-name=srun_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=srun-output/%j.out
#SBATCH --error=srun-output/%j.err

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

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

# Print some information about the job
echo "Running on $(hostname)"
echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Run the distributed job directly with srun
srun python distributed_sum.py --backend=$BACKEND
