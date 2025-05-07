#!/bin/bash
#SBATCH --job-name=ray_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
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

# No modules needed for this example

# Print some information about the job
echo "Running on $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Start Ray head node
export RAY_ADDRESS="$(hostname):6379"
ray start --head --port=6379 --num-cpus=0 --dashboard-host=0.0.0.0

# Have other nodes join the Ray cluster
if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
  # Get the list of worker nodes
  WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $(hostname))
  
  # Start Ray on worker nodes
  for node in $WORKER_NODES; do
    srun --nodes=1 --ntasks=1 -w $node ray start --address=$RAY_ADDRESS --num-cpus=0
  done
fi

# Run the Ray script
python ray_example.py --backend=$BACKEND

# Stop Ray
ray stop
