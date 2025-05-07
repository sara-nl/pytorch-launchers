# PyTorch Launchers Tutorial for SLURM Clusters

This repository provides a practical guide to using various PyTorch launchers within a SLURM environment. It focuses on simplicity and education, with copy-paste ready examples for each launcher.

## Overview

When running distributed PyTorch training on a SLURM cluster, you have several options for launching your jobs:

1. **torchrun**: PyTorch's built-in distributed training launcher
2. **srun**: SLURM's native job launcher
3. **accelerate launch**: Hugging Face's launcher for distributed training
4. **DeepSpeed launcher**: Microsoft's launcher for distributed training with DeepSpeed
5. **Submitit**: Facebook's launcher for submitting jobs to SLURM
6. **Ray**: Ray's distributed computing framework

Each launcher has its own strengths and use cases. This guide will help you understand when and how to use each one.

## Repository Structure

```
pytorch-launchers-tutorial/
│
├── examples/            # Example scripts for each launcher
│   ├── distributed_sum.py  # The main distributed script used by all launchers
│   ├── torchrun.slurm      # SLURM script for torchrun
│   ├── srun.slurm          # SLURM script for srun
│   ├── accelerate.slurm    # SLURM script for accelerate
│   ├── deepspeed.slurm     # SLURM script for DeepSpeed
│   ├── submitit_example.py # Example for Submitit
│   └── ray_example.py      # Example for Ray
│
└── tests/               # Tests for each launcher
    └── test_launchers.py   # Pytest file for testing all launchers
```

## Distributed Script

All launchers will run the same simple distributed script (`distributed_sum.py`) that:
1. Initializes the distributed environment
2. Performs a distributed all-reduce sum operation
3. Verifies the result is correct

This allows for direct comparison between launchers.

## 1. torchrun

### Overview

`torchrun` is PyTorch's built-in distributed training launcher. When used within SLURM, you typically use `srun` to launch `torchrun` across allocated nodes.

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=torchrun_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Get SLURM variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=4

# Run torchrun with srun
srun torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  examples/distributed_sum.py \
  --backend=nccl
```

### Key Points

- Uses `srun` to launch `torchrun` across all allocated nodes
- Sets up rendezvous parameters using SLURM environment variables
- Works well when you want to use PyTorch's built-in distributed features

## 2. srun (Direct)

### Overview

Using `srun` directly is the most straightforward approach on SLURM clusters. PyTorch can detect SLURM environment variables to set up the distributed environment.

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=srun_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

# Run the distributed job directly with srun
srun python examples/distributed_sum.py --backend=nccl
```

### Key Points

- Uses SLURM's `srun` to launch the script directly
- Sets up environment variables manually
- PyTorch can detect SLURM environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.)
- Simple and efficient for SLURM environments

## 3. Accelerate Launch

### Overview

Hugging Face's Accelerate provides a user-friendly interface for distributed training and works well with SLURM.

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=accelerate_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Get SLURM variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_PROCESSES_PER_NODE=4

# Run with accelerate
srun accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_PROCESSES_PER_NODE \
  --num_machines=$NUM_NODES \
  --machine_rank=$SLURM_NODEID \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  examples/distributed_sum.py \
  --backend=nccl
```

### Key Points

- Uses `srun` to launch `accelerate` across all nodes
- Passes SLURM information to Accelerate
- Integrates well with Hugging Face's ecosystem
- Provides a consistent interface across different hardware setups

## 4. DeepSpeed

### Overview

DeepSpeed optimizes training for large models and works well with SLURM for distributed training.

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Create hostfile for DeepSpeed
HOSTFILE=hostfile_$SLURM_JOB_ID.txt
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r host; do
  echo "$host slots=4" >> $HOSTFILE
done

# Run with DeepSpeed
srun deepspeed \
  --hostfile=$HOSTFILE \
  examples/distributed_sum.py \
  --deepspeed \
  --deepspeed_config=examples/ds_config.json
module load deepspeed

# Create hostfile for DeepSpeed
HOSTFILE=hostfile_$SLURM_JOB_ID.txt
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r host; do
  echo "$host slots=4" >> $HOSTFILE
done

# Run with DeepSpeed
deepspeed \
  --hostfile=$HOSTFILE \
  examples/distributed_sum.py \
  --deepspeed \
  --deepspeed_config=examples/ds_config.json
```

### DeepSpeed Config (ds_config.json)

```json
{
  "train_batch_size": 16,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1
  }
}
```

### Key Points

- Creates a hostfile from SLURM allocation
- Uses DeepSpeed's launcher with the hostfile
- Configuration file specifies optimization parameters
- Great for large models that benefit from ZeRO optimization

## 5. Submitit

### Overview

Submitit is a Python package that provides a convenient interface for submitting jobs to SLURM from Python code.

### Python Script

```python
import submitit
import os

# Define the function to be executed
def train_function(backend):
    import sys
    import os
    
    # Get SLURM environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    # Run the distributed script
    cmd = f"python examples/distributed_sum.py --rank={rank} --world-size={world_size} --backend={backend}"
    return os.system(cmd)

# Create a submitit executor
executor = submitit.AutoExecutor(folder="submitit_logs")

# Set SLURM parameters
executor.update_parameters(
    name="submitit_example",
    nodes=2,
    tasks_per_node=4,
    gpus_per_node=4,
    time=10,  # minutes
)

# Submit the job
job = executor.submit(train_function, "nccl")
print(f"Submitted job with ID: {job.job_id}")

# Wait for completion and get result (optional)
result = job.result()
print(f"Job completed with result: {result}")
```

### Key Points

- Submits SLURM jobs directly from Python code
- Allows for dynamic job submission and parameter adjustment
- Can wait for job completion and retrieve results
- Useful for workflows that require programmatic job submission

## 6. Ray

### Overview

Ray provides a distributed computing framework that can work with SLURM for distributed PyTorch training.

### Python Script

```python
import ray
import os
import sys

# Initialize Ray on SLURM
ray.init(address="auto")

@ray.remote(num_gpus=1)
def train_on_gpu(rank, world_size, backend):
    # Set environment variables for PyTorch distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("SLURM_SUBMIT_HOST", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29500"
    
    # Run the distributed script
    cmd = f"python examples/distributed_sum.py --rank={rank} --world-size={world_size} --backend={backend}"
    return os.system(cmd)

# Get SLURM information
world_size = int(os.environ.get("SLURM_NTASKS", 1))

# Launch distributed training
futures = [train_on_gpu.remote(i, world_size, "nccl") for i in range(world_size)]
results = ray.get(futures)

# Check results
success = all(result == 0 for result in results)
print(f"Training {'succeeded' if success else 'failed'}")
sys.exit(0 if success else 1)
```

### SLURM Script for Ray

```bash
#!/bin/bash
#SBATCH --job-name=ray_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00

# Start Ray cluster on SLURM
ray start --head --port=6379 --num-cpus=0

# Other nodes join the Ray cluster
srun --nodes=1-$(($SLURM_JOB_NUM_NODES-1)) --ntasks=1 ray start --address=$HOSTNAME:6379 --num-cpus=0

# Run the Ray script
python examples/ray_example.py

# Stop Ray
ray stop
```

### Key Points

- Initializes a Ray cluster on SLURM allocation
- Uses Ray's task-based parallelism for distributed training
- More flexible than other launchers for complex workflows
- Can be combined with other Ray features (e.g., hyperparameter tuning)

## Testing

To test all launchers locally (simulating a SLURM environment), run:

```bash
pytest tests/test_launchers.py -v
```
