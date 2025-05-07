#!/usr/bin/env python3
"""
Universal local launcher script for running distributed PyTorch
This script can run any of the launchers on a single node, with better readability and Pythonic argument handling.
"""
import argparse
import os
import subprocess
import sys
import torch

LAUNCHERS = ["torchrun", "accelerate", "deepspeed", "multiprocessing"]

def detect_num_processes():
    # Prefer GPUs if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs")
        return num_gpus, "nccl"
    else:
        # Otherwise use number of CPU cores
        num_cpus = len(os.sched_getaffinity(0))
        print(f"No GPUs detected, using {num_cpus} CPU cores")
        return num_cpus, "gloo"

def main():
    parser = argparse.ArgumentParser(description="Universal local launcher for distributed PyTorch")
    parser.add_argument("--launcher", choices=LAUNCHERS, default="torchrun", help="Which launcher to use")
    parser.add_argument("--nproc", type=int, default=0, help="Number of processes (0 = auto-detect)")
    parser.add_argument("--backend", type=str, default=None, choices=["nccl", "gloo"], help="Communication backend (default: auto-detect)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra arguments to pass to the script/launcher")
    args = parser.parse_args()

    # Auto-detect processes and backend if not specified
    num_procs = args.nproc
    backend = args.backend
    if num_procs == 0 or backend is None:
        detected_nproc, detected_backend = detect_num_processes()
        if num_procs == 0:
            num_procs = detected_nproc
        if backend is None:
            backend = detected_backend

    backend_arg = f"--backend={backend}" if backend else ""
    extra_args = args.extra if args.extra else []

    if args.launcher == "torchrun":
        # torchrun does not accept --local; only pass backend and extra args
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_procs}",
            "distributed_sum.py",
            backend_arg,
            *extra_args
        ]
    elif args.launcher == "accelerate":
        # accelerate launch does not accept --local; only pass backend and extra args
        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            f"--num_processes={num_procs}",
            "distributed_sum.py",
            backend_arg,
            *extra_args
        ]
    elif args.launcher == "deepspeed":
        # For single-node, do not use a hostfile (avoids ssh issues)
        cmd = [
            "deepspeed", "distributed_sum.py", backend_arg, "--deepspeed", *extra_args
        ]
    elif args.launcher == "multiprocessing":
        cmd = [
            sys.executable, "distributed_sum.py", f"--world-size={num_procs}", backend_arg, *extra_args
        ]
    else:
        print(f"Unknown launcher: {args.launcher}")
        print(f"Supported launchers: {', '.join(LAUNCHERS)}")
        sys.exit(1)

    # Remove empty arguments
    cmd = [c for c in cmd if c]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {args.launcher} example completed successfully!")
    except subprocess.CalledProcessError:
        print(f"❌ {args.launcher} example failed!")
        sys.exit(1)
    finally:
        # No hostfile cleanup needed since we do not create one for local runs
        pass

if __name__ == "__main__":
    main()
