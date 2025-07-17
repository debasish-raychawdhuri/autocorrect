#!/usr/bin/env python3
"""
Benchmark script to compare the performance of the original memory-optimized implementation
with the new parallel processing implementation.
"""

import argparse
import time
import os
import sys
import torch
import psutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# Import both implementations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import run_3_memory_optimized as memory_opt
import run_3_parallel_processing as parallel_proc

def benchmark_data_loading(data_path, w2v_model, batch_size, num_workers_list):
    """Benchmark data loading performance for different numbers of workers"""
    char_to_id, _ = memory_opt.create_charmap()
    
    # Load dataset once to ensure it's cached
    print("Preloading dataset to ensure fair comparison...")
    dataset = memory_opt.CharGenMemoryDataset(
        data_path, w2v_model, char_to_id,
        ctx_len=10, max_word_len=50, max_gen_len=50
    )
    
    # Benchmark standard DataLoader
    print("\n--- Benchmarking standard DataLoader ---")
    std_times = []
    for num_workers in num_workers_list:
        print(f"\nTesting with {num_workers} workers:")
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Measure time to iterate through 100 batches
        start_time = time.time()
        for i, _ in enumerate(tqdm(dataloader, total=100)):
            if i >= 100:
                break
        end_time = time.time()
        elapsed = end_time - start_time
        std_times.append(elapsed)
        print(f"Time to process 100 batches: {elapsed:.2f} seconds")
        
        # Force cleanup
        del dataloader
        torch.cuda.empty_cache()
    
    # Benchmark parallel processor
    print("\n--- Benchmarking Parallel Processor ---")
    parallel_times = []
    for num_workers in num_workers_list:
        print(f"\nTesting with {num_workers} workers:")
        processor = parallel_proc.ParallelBatchProcessor(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            max_queue_size=num_workers * 2
        )
        
        # Measure time to get 100 batches
        start_time = time.time()
        batches_processed = 0
        with tqdm(total=100) as pbar:
            while batches_processed < 100:
                batch = processor.get_batch(timeout=10.0)
                if batch is not None:
                    batches_processed += 1
                    pbar.update(1)
        end_time = time.time()
        elapsed = end_time - start_time
        parallel_times.append(elapsed)
        print(f"Time to process 100 batches: {elapsed:.2f} seconds")
        
        # Cleanup
        processor.shutdown()
        torch.cuda.empty_cache()
    
    return std_times, parallel_times, num_workers_list

def benchmark_training(data_path, w2v_model, batch_size, num_workers, num_batches=100):
    """Benchmark training performance for both implementations"""
    char_to_id, id_to_char = memory_opt.create_charmap()
    char_vocab_size = len(char_to_id)
    
    # Create dataset
    dataset = memory_opt.CharGenMemoryDataset(
        data_path, w2v_model, char_to_id,
        ctx_len=10, max_word_len=50, max_gen_len=50
    )
    
    # Create model
    context_dim = 10 * w2v_model.vector_size
    word_onehot_dim = 50 * char_vocab_size
    gen_onehot_dim = 50 * char_vocab_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Benchmark standard training
    print("\n--- Benchmarking Standard Training ---")
    model_std = memory_opt.ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)
    
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=1e-5)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    model_std.train()
    start_time = time.time()
    
    for i, (context_vec, misspelled_oh, prefix_oh, next_id) in enumerate(tqdm(dataloader, total=num_batches)):
        if i >= num_batches:
            break
            
        # Move data to device
        context_vec = context_vec.to(device, non_blocking=True)
        misspelled_oh = misspelled_oh.to(device, non_blocking=True)
        prefix_oh = prefix_oh.to(device, non_blocking=True)
        next_id = next_id.to(device, non_blocking=True)
        
        # Forward pass
        logits = model_std(context_vec, misspelled_oh, prefix_oh)
        loss = torch.nn.functional.cross_entropy(logits, next_id)
        
        # Backward pass
        optimizer_std.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_std.parameters(), 1.0)
        optimizer_std.step()
    
    std_time = time.time() - start_time
    print(f"Standard training time for {num_batches} batches: {std_time:.2f} seconds")
    
    # Cleanup
    del model_std, optimizer_std, dataloader
    torch.cuda.empty_cache()
    
    # Benchmark parallel training
    print("\n--- Benchmarking Parallel Training ---")
    model_parallel = parallel_proc.ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)
    
    optimizer_parallel = torch.optim.Adam(model_parallel.parameters(), lr=1e-5)
    
    processor = parallel_proc.ParallelBatchProcessor(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_queue_size=num_workers * 2
    )
    
    model_parallel.train()
    start_time = time.time()
    
    batches_processed = 0
    with tqdm(total=num_batches) as pbar:
        while batches_processed < num_batches:
            batch = processor.get_batch(timeout=10.0)
            if batch is None:
                time.sleep(0.1)
                continue
                
            # Unpack batch
            context_vec, misspelled_oh, prefix_oh, next_id = batch
            
            # Move data to device
            context_vec = context_vec.to(device, non_blocking=True)
            misspelled_oh = misspelled_oh.to(device, non_blocking=True)
            prefix_oh = prefix_oh.to(device, non_blocking=True)
            next_id = next_id.to(device, non_blocking=True)
            
            # Forward pass
            logits = model_parallel(context_vec, misspelled_oh, prefix_oh)
            loss = torch.nn.functional.cross_entropy(logits, next_id)
            
            # Backward pass
            optimizer_parallel.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_parallel.parameters(), 1.0)
            optimizer_parallel.step()
            
            batches_processed += 1
            pbar.update(1)
    
    parallel_time = time.time() - start_time
    print(f"Parallel training time for {num_batches} batches: {parallel_time:.2f} seconds")
    
    # Cleanup
    processor.shutdown()
    del model_parallel, optimizer_parallel
    torch.cuda.empty_cache()
    
    return std_time, parallel_time

def plot_results(std_times, parallel_times, num_workers_list, output_path="benchmark_results.png"):
    """Plot benchmark results"""
    plt.figure(figsize=(12, 8))
    
    # Data loading comparison
    plt.subplot(2, 1, 1)
    plt.plot(num_workers_list, std_times, 'o-', label='Standard DataLoader')
    plt.plot(num_workers_list, parallel_times, 's-', label='Parallel Processor')
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.title('Data Loading Performance: Time to Process 100 Batches')
    plt.grid(True)
    plt.legend()
    
    # Calculate speedup
    speedups = [s/p for s, p in zip(std_times, parallel_times)]
    
    # Speedup plot
    plt.subplot(2, 1, 2)
    plt.plot(num_workers_list, speedups, 'o-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup (Standard / Parallel)')
    plt.title('Speedup Factor of Parallel Processing vs Standard DataLoader')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel processing performance")
    parser.add_argument("--data", type=str, required=True, help="Path to optimized data file")
    parser.add_argument("--word2vec", type=str, required=True, help="Path to word2vec model")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--max_workers", type=int, default=48, help="Maximum number of workers to test")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Output path for benchmark results")
    args = parser.parse_args()
    
    # Print system information
    print("System Information:")
    mem_info = psutil.virtual_memory()
    print(f"- Memory: {mem_info.total / (1024**3):.1f} GB total, {mem_info.available / (1024**3):.1f} GB available")
    print(f"- CPU cores: {os.cpu_count()}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
    
    # Load word2vec model
    print(f"Loading word2vec model from {args.word2vec}...")
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    
    # Generate worker counts to test
    worker_counts = [1, 2, 4, 8, 16, 24, 32]
    if args.max_workers > 32:
        worker_counts.append(args.max_workers)
    worker_counts = [w for w in worker_counts if w <= args.max_workers]
    
    # Benchmark data loading
    print("\n=== Benchmarking Data Loading ===")
    std_times, parallel_times, workers = benchmark_data_loading(
        args.data, w2v_model, args.batch_size, worker_counts
    )
    
    # Plot results
    plot_results(std_times, parallel_times, workers, args.output)
    
    # Benchmark training with optimal worker count
    # Find optimal worker count based on speedup
    speedups = [s/p for s, p in zip(std_times, parallel_times)]
    optimal_idx = speedups.index(max(speedups))
    optimal_workers = worker_counts[optimal_idx]
    
    print(f"\n=== Benchmarking Training with {optimal_workers} workers ===")
    std_train_time, parallel_train_time = benchmark_training(
        args.data, w2v_model, args.batch_size, optimal_workers
    )
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Optimal worker count: {optimal_workers}")
    print(f"Data loading speedup with {optimal_workers} workers: {speedups[optimal_idx]:.2f}x")
    print(f"Training speedup: {std_train_time / parallel_train_time:.2f}x")
    
    # Save summary to file
    with open("benchmark_summary.txt", "w") as f:
        f.write("=== Benchmark Summary ===\n")
        f.write(f"System: {os.cpu_count()} CPU cores, {mem_info.total / (1024**3):.1f} GB RAM\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n\n")
        
        f.write("Data Loading Performance:\n")
        for i, workers in enumerate(worker_counts):
            f.write(f"- {workers} workers: Standard={std_times[i]:.2f}s, Parallel={parallel_times[i]:.2f}s, Speedup={speedups[i]:.2f}x\n")
        
        f.write(f"\nOptimal worker count: {optimal_workers}\n")
        f.write(f"Training performance with {optimal_workers} workers:\n")
        f.write(f"- Standard: {std_train_time:.2f}s\n")
        f.write(f"- Parallel: {parallel_train_time:.2f}s\n")
        f.write(f"- Speedup: {std_train_time / parallel_train_time:.2f}x\n")
    
    print(f"Summary saved to benchmark_summary.txt")

if __name__ == "__main__":
    main()
