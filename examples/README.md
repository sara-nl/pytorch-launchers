# PyTorch Launchers Examples

This directory contains example scripts for running distributed PyTorch training using various launchers, both locally and on SLURM clusters.

## Quick Start

### Local Execution

To run any of the launchers locally on a single machine, use the `run_local.sh` script:

```bash
# Run with torchrun (default)
./run_local.sh

# Run with a specific launcher
./run_local.sh --launcher torchrun
./run_local.sh --launcher accelerate
./run_local.sh --launcher deepspeed
./run_local.sh --launcher multiprocessing

# Specify number of processes
./run_local.sh --launcher torchrun --nproc 4

# Specify backend (nccl or gloo)
./run_local.sh --launcher torchrun --backend nccl
```

The script will auto-detect the number of GPUs (or CPU cores if no GPUs are available) and choose the appropriate backend (NCCL for GPUs, Gloo for CPU-only).

### SLURM Execution

To run on a SLURM cluster, use the appropriate `.slurm` script:

```bash
# Submit a job using torchrun
sbatch torchrun.slurm

# Submit a job using direct srun
sbatch srun.slurm

# Submit a job using Accelerate
sbatch accelerate.slurm

# Submit a job using DeepSpeed
sbatch deepspeed.slurm

# Submit a job using Ray
sbatch ray.slurm
```

For Submitit, you can use the Python script directly:

```bash
# Submit a job using Submitit
python submitit_example.py

# Run Submitit locally for testing
python submitit_example.py --local
```

## Distributed Script

All launchers use the same `distributed_sum.py` script, which performs a simple distributed all-reduce sum operation. The script can detect the distributed environment automatically and supports both SLURM and local execution.

```bash
# Run directly with auto-detection
python distributed_sum.py

# Force local execution mode
python distributed_sum.py --local

# Specify backend manually
python distributed_sum.py --backend nccl
```

## Customization

You can customize the SLURM scripts for your specific cluster by modifying:

1. The SLURM directives at the top of each script (`#SBATCH` lines)
2. The number of nodes, tasks, and GPUs
3. Any module loading commands if needed for your environment

## Troubleshooting

If you encounter issues:

1. **Backend selection**: Make sure you're using the right backend for your hardware (NCCL for GPUs, Gloo for CPU)
2. **Port conflicts**: If you get "address already in use" errors, try a different port number
3. **SLURM environment**: Check that your SLURM environment variables are correctly set
4. **DeepSpeed issues**: DeepSpeed requires CUDA to be properly set up
5. **Ray cluster**: Make sure Ray can start properly on your SLURM allocation
