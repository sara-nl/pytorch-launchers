#!/usr/bin/env python3
"""
A simple distributed PyTorch script that performs an all-reduce sum operation.
This script is designed to work with all launchers and detect SLURM environment variables.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist


def get_default_backend():
    """
    Determine the best backend based on available hardware.
    Use NCCL if GPUs are available, otherwise use Gloo.
    """
    if torch.cuda.is_available():
        return 'nccl'
    else:
        return 'gloo'

def get_slurm_environment_ranks():
    """
    Detect SLURM environment variables and return rank, world_size, and local_rank.
    """
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    return rank, world_size, local_rank

def setup_distributed_env():
    """
    Set up the distributed environment by detecting environment variables
    from various launchers (SLURM, torchrun, etc.)
    """

        
    # Check for SLURM environment variables
    if 'SLURM_PROCID' in os.environ:
        print("Detected SLURM environment variables")
        rank, world_size, local_rank = get_slurm_environment_ranks()
    
    # Check for torchrun/torch.distributed.launch environment variables
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("Detected torchrun/torch.distributed.launch environment variables")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    
    # Check for DeepSpeed environment variables
    elif 'DLTS_HOSTFILE' in os.environ or 'DLTS_RANK' in os.environ:
        print("Detected DeepSpeed environment variables")
        rank, world_size, local_rank = get_deepspeed_environment_ranks()
    
    else:
        print("No distributed environment detected")
        rank = 0
        world_size = 1
        local_rank = 0

    print(f"rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        
def get_deepspeed_environment_ranks():
    """
    Detect DeepSpeed environment variables and return rank, world_size, and local_rank.
    """
    rank = int(os.environ.get('DLTS_RANK', os.environ.get('RANK', '0')))
    world_size = int(os.environ.get('DLTS_NUM_PROCS', os.environ.get('WORLD_SIZE', '1')))
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    return rank, world_size, local_rank


def init_distributed(rank, world_size, backend):
    """
    Initialize the distributed process group.
    """
    # Set environment variables if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"Initialized process {rank} / {world_size} with backend: {backend}")


def run_distributed_sum(rank, world_size, backend):
    """
    Run a simple distributed sum operation.
    """
    # Initialize the distributed environment
    init_distributed(rank, world_size, backend)
    
    # Pick device based on backend
    if backend == "nccl" and torch.cuda.is_available():
        # Use the correct GPU for each process
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    else:
        device = torch.device("cpu")

    # Create a random tensor on the correct device
    tensor = torch.ones(10, device=device) * (rank + 1)
    print(f"Rank {rank}: Initial tensor: {tensor}")

    # Calculate the expected sum on the same device
    expected_sum = sum([(i + 1) for i in range(world_size)]) * torch.ones(10, device=device)
    
    # Perform an all-reduce sum operation
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: Reduced tensor: {tensor}")
    
    # Verify the result
    is_correct = torch.allclose(tensor, expected_sum)
    print(f"Rank {rank}: Result is correct: {is_correct}")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return is_correct


def detect_local_world_size():
    """
    Detect the number of GPUs available locally for automatic world size determination.
    If no GPUs are available, default to the number of CPU cores.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        import multiprocessing
        return multiprocessing.cpu_count()

def main():
    """
    Main function to parse arguments and run the distributed sum.
    """
    parser = argparse.ArgumentParser(description='Distributed Sum Example')
    parser.add_argument('--rank', type=int, default=-1,
                        help='Rank of the current process (default: auto-detect)')
    parser.add_argument('--world-size', type=int, default=-1,
                        help='Number of processes (default: auto-detect)')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['nccl', 'gloo'],
                        help='Distributed backend to use (default: auto-detect)')
    parser.add_argument('--local', action='store_true',
                        help='Force local execution mode (ignore distributed environment)')
    parser.add_argument('--deepspeed', action='store_true',
                        help='Running with DeepSpeed')
    args = parser.parse_args()
    
    # Choose default backend based on available hardware
    if args.backend is None:
        args.backend = get_default_backend()
        print(f"Auto-selected backend: {args.backend}")
    
    # Auto-detect distributed environment if not specified
    if args.local:
        # Force local execution mode
        args.world_size = detect_local_world_size() if args.world_size == -1 else args.world_size
        args.rank = 0 if args.rank == -1 else args.rank
        args.local_rank = 0 if args.local_rank == -1 else args.local_rank
        print(f"Running in local mode: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")
    else:
        # Auto-detect distributed environment
        for arg_name, default_val in [('rank', args.rank), ('world_size', args.world_size), ('local_rank', args.local_rank)]:
            if default_val == -1:
                if args.local:
                    default_val = detect_local_world_size() if arg_name == 'world_size' else 0
                else:
                    default_val = setup_distributed_env()[['rank', 'world_size', 'local_rank'].index(arg_name)]
                setattr(args, arg_name, default_val)
    # Print basic environment info
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device_name = torch.cuda.get_device_name(args.local_rank)
    else:
        device_name = 'CPU'
    
    print(f"Running on: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Process rank: {args.rank}, World size: {args.world_size}, Local rank: {args.local_rank}")
    print(f"Using backend: {args.backend}")
    
    # Run the distributed sum
    success = run_distributed_sum(args.rank, args.world_size, args.backend)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
