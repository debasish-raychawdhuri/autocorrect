#!/usr/bin/env python3
"""
Helper script to launch distributed training on multiple GPUs.
This script uses torch.distributed.launch to start multiple processes.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch distributed training for run_reu.py")
    parser.add_argument("--nproc_per_node", type=int, default=None, 
                        help="Number of processes per node (defaults to number of GPUs)")
    parser.add_argument("--data", type=str, default="autogen_char_data.json",
                        help="Path to training data")
    parser.add_argument("--word2vec", type=str, required=True,
                        help="Path to word2vec model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--model", type=str, default="char_autocorrect.pt",
                        help="Path to save model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more verbose output")
    args = parser.parse_args()

    # Get number of available GPUs
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPUs available. Exiting.")
        sys.exit(1)
    
    # Use all available GPUs if not specified
    nproc = args.nproc_per_node if args.nproc_per_node is not None else num_gpus
    print(f"Launching distributed training on {nproc} GPUs")
    
    # Check GPU memory before launching
    print("GPU Memory Status:")
    for i in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        print(f"  GPU {i}: {total_memory / (1024**3):.1f} GB total memory")

    # Build command for torchrun (replaces deprecated torch.distributed.launch)
    cmd = [
        sys.executable, 
        "-m", 
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port=29500",  # Specify port to avoid conflicts
        "--standalone",  # Single node training
        "run_reu.py",
        "--train",
        "--distributed",
        f"--data={args.data}",
        f"--word2vec={args.word2vec}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--model={args.model}"
    ]

    # Launch the distributed training with full error output
    print(f"Running command: {' '.join(cmd)}")
    
    # Set environment variables to get more detailed error information
    env = os.environ.copy()
    
    # Always set these for better debugging
    env["CUDA_LAUNCH_BLOCKING"] = "1"          # Synchronous CUDA kernel launches
    env["TORCH_USE_CUDA_DSA"] = "1"            # Enable device-side assertions
    env["PYTHONFAULTHANDLER"] = "1"           # Enable Python fault handler
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"     # Better NCCL error handling
    
    if args.debug:
        env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Enable detailed distributed debugging
        env["NCCL_DEBUG"] = "INFO"                # Enable NCCL debugging
        env["NCCL_DEBUG_SUBSYS"] = "ALL"          # Debug all NCCL subsystems
    
    # Memory management
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Limit memory fragmentation
    
    # Run the process with output streaming to console
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Training failed with return code {return_code}")
        sys.exit(return_code)
    
    print("Distributed training completed successfully!")

if __name__ == "__main__":
    main()
