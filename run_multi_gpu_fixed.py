#!/usr/bin/env python3
"""
Multi-GPU training script with fixed parallel data loading
"""
import torch
import torch.nn as nn
import sys
import os
import json
import time
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need to avoid global device setup
from run_reu import (
    create_charmap, pad_context, vectorize_context, one_hot_chars, 
    ResNetFFN, train_model
)

class FastParallelDataset(Dataset):
    """Dataset that actually uses parallel processing for data loading"""
    
    def __init__(self, json_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50, num_workers=4):
        self.json_path = json_path
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len
        
        print(f"Building dataset index with {num_workers} workers...")
        start_time = time.time()
        
        # Build offsets using actual parallel processing
        self.offsets = self._build_offsets_parallel(num_workers)
        
        end_time = time.time()
        print(f"Dataset index built with {len(self.offsets)} samples in {end_time - start_time:.2f} seconds")
    
    def _build_offsets_worker(self, args):
        """Worker function for parallel offset building"""
        json_path, start_pos, end_pos = args
        offsets = []
        
        with open(json_path, 'rb') as f:
            f.seek(start_pos)
            
            # Skip to start of line if not at beginning
            if start_pos > 0:
                f.readline()
                
            pos = f.tell()
            
            while pos < end_pos:
                line = f.readline()
                if not line:
                    break
                offsets.append(pos)
                pos = f.tell()
        
        return offsets
    
    def _build_offsets_parallel(self, num_workers):
        """Build offsets using actual multiprocessing"""
        # Get file size
        file_size = os.path.getsize(self.json_path)
        
        # Split file into chunks for workers
        chunk_size = file_size // num_workers
        tasks = []
        
        for i in range(num_workers):
            start_pos = i * chunk_size
            end_pos = (i + 1) * chunk_size if i < num_workers - 1 else file_size
            tasks.append((self.json_path, start_pos, end_pos))
        
        # Use ProcessPoolExecutor instead of multiprocessing.Pool
        all_offsets = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._build_offsets_worker, tasks))
            
        # Combine results
        for result in results:
            all_offsets.extend(result)
            
        return sorted(all_offsets)
    
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, idx):
        with open(self.json_path, encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line.strip())
        
        context = pad_context(sample["context"], self.ctx_len)
        misspelled = sample["misspelled"]
        prefix = sample["generated_prefix"]
        next_char = sample["next_char"]
        context_vec = vectorize_context(context, self.w2v_model, self.ctx_len)
        misspelled_oh = one_hot_chars(misspelled, self.char_to_id, self.max_word_len)
        prefix_oh = one_hot_chars(prefix, self.char_to_id, self.max_gen_len)
        
        # "<eow>" is used as end-of-word
        if next_char == "<eow>":
            next_id = self.char_to_id["<eow>"]
        else:
            next_id = self.char_to_id.get(next_char, 0)
        
        return (
            torch.tensor(context_vec, dtype=torch.float32),
            torch.tensor(misspelled_oh, dtype=torch.float32),
            torch.tensor(prefix_oh, dtype=torch.float32),
            torch.tensor(next_id, dtype=torch.long)
        )

def create_optimized_dataloader(dataset, batch_size, num_workers):
    """Create DataLoader with optimal settings for parallel processing"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4,  # Increased prefetch
        drop_last=True,
        multiprocessing_context=mp.get_context('spawn')  # Use spawn instead of fork
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="autogen_char_data.json")
    parser.add_argument("--word2vec", type=str, required=True, help="Custom word2vec file (.npz, .pkl, .json)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="char_autocorrect.pt")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus < 1:
        print("No GPUs available. Exiting.")
        sys.exit(1)
    
    # Use first GPU as primary
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    # Set worker count if not specified
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 16)  # Cap at 16 workers for stability
    
    print(f"Using {args.num_workers} worker processes")
    
    # Load word2vec model
    from custom_word2vec import load_custom_word2vec
    w2v_model = load_custom_word2vec(args.word2vec)
    char_to_id, id_to_char = create_charmap()
    char_vocab_size = len(char_to_id)

    context_dim = args.ctx_len * w2v_model.vector_size
    word_onehot_dim = args.max_word_len * char_vocab_size
    gen_onehot_dim = args.max_gen_len * char_vocab_size

    # Create model
    model = ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Use DataParallel if multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel for {num_gpus} GPUs")
    
    # Load existing model if available
    from pathlib import Path
    model_path = args.model
    if Path(model_path).exists():
        print(f"Loading existing model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]
            
            # Handle DataParallel state dict
            if num_gpus > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            elif num_gpus == 1 and any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            
            model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch")
    
    # Create dataset with parallel processing
    print("Creating dataset...")
    dataset = FastParallelDataset(
        args.data, 
        w2v_model, 
        char_to_id,
        ctx_len=args.ctx_len, 
        max_word_len=args.max_word_len, 
        max_gen_len=args.max_gen_len,
        num_workers=args.num_workers
    )
    
    # Create optimized dataloader
    dataloader = create_optimized_dataloader(dataset, args.batch_size, args.num_workers)
    
    print(f"DataLoader created with {len(dataset)} samples")
    
    # Train model
    train_model(
        model, 
        dataloader, 
        vocab_size=char_vocab_size, 
        epochs=args.epochs, 
        save_path=args.model,
        local_rank=0,
        is_distributed=False,
        num_workers=args.num_workers
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()