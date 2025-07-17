#!/usr/bin/env python3
"""
Multi-GPU training script using DataParallel to bypass CUDA driver issues
"""
import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_reu import *

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="autogen_char_data.json")
    parser.add_argument("--word2vec", type=str, required=True)
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
        import multiprocessing as mp
        args.num_workers = max(1, min(mp.cpu_count(), 32))  # Cap at 32 workers
    
    print(f"Using {args.num_workers} worker processes")
    
    # Load word2vec model
    from gensim.models import KeyedVectors
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
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
            checkpoint = torch.load(model_path, map_location=device)
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
    
    # Create dataset
    print("Creating dataset...")
    dataset = CharGenLazyDataset(
        args.data, 
        w2v_model, 
        char_to_id,
        ctx_len=args.ctx_len, 
        max_word_len=args.max_word_len, 
        max_gen_len=args.max_gen_len,
        num_workers=args.num_workers
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2,
        drop_last=True
    )
    
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