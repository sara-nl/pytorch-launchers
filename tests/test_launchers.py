#!/usr/bin/env python3
"""
Consolidated test file for all PyTorch launchers.
This file provides tests that simulate each launcher in a local environment.
"""

import os
import sys
import subprocess
import re
import pytest
import shutil
from pathlib import Path


# Get the repository root directory
REPO_ROOT = Path(__file__).parent.parent.absolute()


def setup_module(module):
    """Setup for all tests."""
    print(f"Repository root: {REPO_ROOT}")
    print(f"PyTorch version: {subprocess.getoutput('python -c "import torch; print(torch.__version__)"')}")


def check_output_for_success(stdout, stderr, num_processes):
    """
    Check if the output indicates successful distributed training.
    
    Args:
        stdout: Standard output from the command
        stderr: Standard error from the command
        num_processes: Expected number of processes
        
    Returns:
        True if the test passes, False otherwise
    """
    # Check if all processes initialized correctly
    init_pattern = re.compile(r"Initialized process (\d+) / (\d+)")
    init_matches = init_pattern.findall(stdout)
    
    if len(init_matches) != num_processes:
        print(f"Error: Expected {num_processes} initialized processes, but found {len(init_matches)}")
        return False
    
    # Check if all processes reported correct results
    correct_pattern = re.compile(r"Rank (\d+): Result is correct: True")
    correct_matches = correct_pattern.findall(stdout)
    
    if len(correct_matches) != num_processes:
        print(f"Error: Expected {num_processes} correct results, but found {len(correct_matches)}")
        return False
    
    return True


def run_command(cmd, cwd=None):
    """
    Run a command and return the output.
    
    Args:
        cmd: Command to run (list of strings)
        cwd: Working directory
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    print(f"Running command: {' '.join(cmd)}")
    
    if cwd is None:
        cwd = REPO_ROOT
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    
    stdout, stderr = process.communicate()
    
    print("\nSTDOUT:")
    print(stdout)
    print("\nSTDERR:")
    print(stderr)
    
    return process.returncode, stdout, stderr


@pytest.mark.parametrize("num_processes", [2])
def test_torchrun(num_processes):
    """Test torchrun launcher."""
    # Skip if torchrun is not available
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available")
    
    # Command to run
    cmd = [
        "torchrun",
        "--nproc_per_node", str(num_processes),
        str(REPO_ROOT / "examples" / "distributed_sum.py"),
        "--backend", "gloo"
    ]
    
    # Run the command
    returncode, stdout, stderr = run_command(cmd)
    
    # Check the output
    assert returncode == 0, "torchrun command failed"
    assert check_output_for_success(stdout, stderr, num_processes), "torchrun test failed"


@pytest.mark.parametrize("num_nodes,tasks_per_node", [(1, 2)])
def test_srun_simulation(num_nodes, tasks_per_node):
    """Test SLURM srun simulation."""
    # Calculate total number of tasks
    total_tasks = num_nodes * tasks_per_node
    
    # Set up environment to simulate SLURM
    env = os.environ.copy()
    
    # Launch processes to simulate SLURM
    processes = []
    for node in range(num_nodes):
        for local_rank in range(tasks_per_node):
            # Calculate global rank
            rank = node * tasks_per_node + local_rank
            
            # Set SLURM-like environment variables
            proc_env = env.copy()
            proc_env["SLURM_PROCID"] = str(rank)
            proc_env["SLURM_NTASKS"] = str(total_tasks)
            proc_env["SLURM_LOCALID"] = str(local_rank)
            proc_env["SLURM_NODEID"] = str(node)
            proc_env["MASTER_ADDR"] = "localhost"
            proc_env["MASTER_PORT"] = "12355"
            
            # Command to run
            cmd = [
                sys.executable,
                str(REPO_ROOT / "examples" / "distributed_sum.py"),
                "--backend", "gloo"
            ]
            
            # Launch the process
            print(f"Launching process with rank {rank} (node {node}, local_rank {local_rank})")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=proc_env
            )
            
            processes.append((process, rank))
    
    # Collect output from all processes
    all_stdout = ""
    all_stderr = ""
    
    for process, rank in processes:
        stdout, stderr = process.communicate()
        all_stdout += f"Process {rank} stdout:\n{stdout}\n"
        all_stderr += f"Process {rank} stderr:\n{stderr}\n"
        assert process.returncode == 0, f"Process {rank} failed with return code {process.returncode}"
    
    print("\nALL STDOUT:")
    print(all_stdout)
    print("\nALL STDERR:")
    print(all_stderr)
    
    # Check the output
    assert check_output_for_success(all_stdout, all_stderr, total_tasks), "SLURM simulation test failed"


@pytest.mark.parametrize("num_processes", [2])
def test_accelerate(num_processes):
    """Test accelerate launcher."""
    # Skip if accelerate is not available
    if shutil.which("accelerate") is None:
        pytest.skip("accelerate not available")
    
    # Command to run
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_processes),
        str(REPO_ROOT / "examples" / "distributed_sum.py"),
        "--backend", "gloo"
    ]
    
    # Run the command
    returncode, stdout, stderr = run_command(cmd)
    
    # Check the output
    assert returncode == 0, "accelerate command failed"
    assert check_output_for_success(stdout, stderr, num_processes), "accelerate test failed"


def test_submitit_simulation():
    """Test submitit simulation."""
    # Skip if submitit is not available
    try:
        import submitit
    except ImportError:
        pytest.skip("submitit not available")
    
    # Define a function to simulate a SLURM job
    def train_function():
        # Set up environment to simulate SLURM
        os.environ["SLURM_PROCID"] = "0"
        os.environ["SLURM_NTASKS"] = "1"
        os.environ["SLURM_LOCALID"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Run the distributed script
        cmd = [
            sys.executable,
            str(REPO_ROOT / "examples" / "distributed_sum.py"),
            "--backend", "nccl"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Return the output
        return process.returncode == 0, process.stdout, process.stderr
    
    # Run the function locally
    success, stdout, stderr = train_function()
    
    print("\nSTDOUT:")
    print(stdout)
    print("\nSTDERR:")
    print(stderr)
    
    # Check the output
    assert success, "submitit simulation command failed"
    assert check_output_for_success(stdout, stderr, 1), "submitit simulation test failed"


def test_ray_simulation():
    """Test Ray simulation."""
    # Skip if Ray is not available
    try:
        import ray
    except ImportError:
        pytest.skip("ray not available")
    
    # Initialize Ray locally
    ray.init(local_mode=True)
    
    # Define a Ray task
    @ray.remote
    def train_on_gpu(rank, world_size):
        # Set environment variables for PyTorch distributed
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Run the distributed script
        cmd = [
            sys.executable,
            str(REPO_ROOT / "examples" / "distributed_sum.py"),
            "--backend", "gloo"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Return the output
        return process.returncode == 0, process.stdout, process.stderr
    
    # Run a single Ray task
    success, stdout, stderr = ray.get(train_on_gpu.remote(0, 1))
    
    print("\nSTDOUT:")
    print(stdout)
    print("\nSTDERR:")
    print(stderr)
    
    # Shutdown Ray
    ray.shutdown()
    
    # Check the output
    assert success, "Ray simulation command failed"
    assert check_output_for_success(stdout, stderr, 1), "Ray simulation test failed"


if __name__ == "__main__":
    # Run all tests
    pytest.main(["-xvs", __file__])
