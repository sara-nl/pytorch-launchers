#!/usr/bin/env python3
"""
Example script for distributed PyTorch training using Ray on a SLURM cluster.
This script demonstrates how to use Ray to manage distributed training jobs.
"""

import os
import sys
import argparse
import ray


def parse_args():
    parser = argparse.ArgumentParser(description='Ray SLURM Example')
    parser.add_argument('--world-size', type=int, default=None,
                        help='Total number of processes (default: auto-detect from SLURM)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='PyTorch distributed backend')
    return parser.parse_args()


@ray.remote(num_gpus=1)
def train_on_gpu(rank, world_size, backend):
    """
    Ray task function that runs on a single GPU.
    Sets up the PyTorch distributed environment and runs the distributed_sum.py script.
    """
    import os
    import subprocess
    import torch
    
    # Set environment variables for PyTorch distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("SLURM_SUBMIT_HOST", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29500"
    
    # Get local rank (within the node)
    node_rank = int(os.environ.get("SLURM_NODEID", 0))
    tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    local_rank = rank % tasks_per_node
    
    # Auto-detect backend if not specified
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(f"Auto-selected backend: {backend}")
    
    # Print some information
    print(f"Running on {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
    print(f"Using backend: {backend}")
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the distributed script
    cmd = [
        "python",
        os.path.join(script_dir, "distributed_sum.py"),
        f"--rank={rank}",
        f"--world-size={world_size}",
        f"--local-rank={local_rank}",
        f"--backend={backend}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, check=True)
    
    return process.returncode == 0


def main():
    """
    Main function to initialize Ray and launch distributed training.
    """
    args = parse_args()
    
    # Initialize Ray (will auto-detect SLURM environment)
    ray.init(address="auto")
    
    # Get world size from SLURM if not specified
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    print(f"Starting Ray distributed training with {world_size} processes")
    
    # Launch distributed training
    futures = [train_on_gpu.remote(i, world_size, args.backend) for i in range(world_size)]
    
    # Wait for all tasks to complete
    print("Waiting for all processes to complete...")
    results = ray.get(futures)
    
    # Check results
    success = all(results)
    print(f"Training {'succeeded' if success else 'failed'}")
    
    # Shutdown Ray
    ray.shutdown()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
