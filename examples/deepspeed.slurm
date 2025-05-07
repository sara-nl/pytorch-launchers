#!/bin/bash
#SBATCH --job-name=deepspeed_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
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

# Print some information about the job
echo "Running on $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Create hostfile for DeepSpeed
HOSTFILE=hostfile_$SLURM_JOB_ID.txt
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r host; do
  echo "$host slots=4" >> $HOSTFILE
done

echo "Created hostfile: $HOSTFILE"
cat $HOSTFILE

# Run with DeepSpeed
deepspeed \
  --hostfile=$HOSTFILE \
  distributed_sum.py \
  --deepspeed \
  --deepspeed_config=ds_config.json \
  --backend=$BACKEND

# Clean up hostfile
rm $HOSTFILE
