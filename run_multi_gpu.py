#!/usr/bin/env python3
"""
Helper script to launch multi-GPU training using DataParallel.
This is simpler than distributed training and doesn't require multiple processes.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch multi-GPU training for run_reu.py using DataParallel")
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
    args = parser.parse_args()

    # Build command for multi-GPU training with DataParallel
    cmd = [
        sys.executable,
        "run_reu.py",
        "--train",
        "--multi_gpu",
        f"--data={args.data}",
        f"--word2vec={args.word2vec}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--model={args.model}"
    ]

    # Launch the training
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the process with output streaming to console
    process = subprocess.Popen(
        cmd,
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
    
    print("Multi-GPU training completed successfully!")

if __name__ == "__main__":
    main()
