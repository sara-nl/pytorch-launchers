#!/usr/bin/env python3
"""
Example script for submitting distributed PyTorch jobs to SLURM using Submitit.
This script demonstrates how to use Submitit to submit and manage SLURM jobs.
"""

import os
import sys
import submitit
import argparse


def get_default_backend():
    """
    Determine the best backend based on available hardware.
    Use NCCL if GPUs are available, otherwise use Gloo.
    """
    import torch
    if torch.cuda.is_available():
        return 'nccl'
    else:
        return 'gloo'

def parse_args():
    parser = argparse.ArgumentParser(description='Submitit SLURM Example')
    parser.add_argument('--nodes', type=int, default=2,
                        help='Number of nodes to request')
    parser.add_argument('--tasks-per-node', type=int, default=4,
                        help='Number of tasks per node')
    parser.add_argument('--gpus-per-node', type=int, default=4,
                        help='Number of GPUs per node')
    parser.add_argument('--time', type=int, default=10,
                        help='Job time limit in minutes')
    parser.add_argument('--partition', type=str, default=None,
                        help='SLURM partition to use')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['nccl', 'gloo'],
                        help='PyTorch distributed backend (default: auto-detect)')
    parser.add_argument('--local', action='store_true',
                        help='Run in local mode (for testing)')
    args = parser.parse_args()
    
    # Auto-detect backend if not specified
    if args.backend is None:
        args.backend = get_default_backend()
        print(f"Auto-selected backend: {args.backend}")
    
    return args


def train_function(backend, local_mode=False):
    """
    Function that will be executed on the SLURM cluster or locally.
    This function sets up the distributed environment and runs the distributed_sum.py script.
    """
    import os
    import subprocess
    import torch
    
    # Auto-detect backend if not specified
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(f"Auto-selected backend: {backend}")
    
    if local_mode:
        # For local testing, simulate a single-node environment
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        # Get SLURM environment variables
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        
        # Set master address and port
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_SUBMIT_HOST", "127.0.0.1")
        os.environ["MASTER_PORT"] = "29500"
    
    # Print some information
    print(f"Running on {os.environ.get('SLURMD_NODENAME', 'unknown') if not local_mode else 'local machine'}")
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
    
    if local_mode:
        cmd.append("--local")
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, check=True)
    
    return process.returncode == 0


def main():
    """
    Main function to submit the job to SLURM using Submitit or run locally.
    """
    args = parse_args()
    
    if args.local:
        # Run locally for testing
        print("Running in local mode for testing")
        result = train_function(args.backend, local_mode=True)
        print(f"Local run completed with result: {result}")
        return 0 if result else 1
    
    # Create a submitit executor
    executor = submitit.AutoExecutor(folder="submitit_logs")
    
    # Set SLURM parameters
    slurm_kwargs = {
        "nodes": args.nodes,
        "ntasks_per_node": args.tasks_per_node,
        "gpus_per_node": args.gpus_per_node,
        "time": args.time,  # minutes
    }
    
    if args.partition:
        slurm_kwargs["slurm_partition"] = args.partition
    
    executor.update_parameters(
        name="submitit_example",
        **slurm_kwargs
    )
    
    # Submit the job
    job = executor.submit(train_function, args.backend, False)
    print(f"Submitted job with ID: {job.job_id}")
    
    # Wait for completion and get result (optional)
    print("Waiting for job to complete...")
    result = job.result()
    print(f"Job completed with result: {result}")
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
