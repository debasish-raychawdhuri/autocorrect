import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import string
import os
import time

# Set up device
def setup_device(gpu_id=None, use_multi_gpu=False):
    if not torch.cuda.is_available():
        return torch.device("cpu"), 1
    
    if use_multi_gpu:
        # Count available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"✅ Using {num_gpus} GPUs")
            return torch.device("cuda"), num_gpus
        else:
            print(f"✅ Only 1 GPU available, using single GPU mode")
            return torch.device("cuda:0"), 1
    elif gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"✅ Using GPU: {gpu_id}")
        return device, 1
    else:
        device = torch.device("cuda:0")
        print(f"✅ Using single GPU: 0")
        return device, 1

# Default device setup (will be updated in main)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Initial device setup: {device}")

# Custom birely activation function: birely(x) = relu(x) - 0.3 * relu(-x)
def birely(x):
    return F.relu(x) - 0.3 * F.relu(-x)

# ---- Char Map ----

def create_charmap():
    printable = string.printable.replace('"', '')  # Avoid JSON quote issues
    char_list = list(printable) + ["<eow>"]
    char_to_id = {c: i for i, c in enumerate(char_list)}
    id_to_char = {i: c for i, c in enumerate(char_list)}
    return char_to_id, id_to_char

# ---- Input Preparation ----

def one_hot_chars(seq, char_to_id, max_len):
    arr = np.zeros((max_len, len(char_to_id)), dtype=np.float32)
    seq = seq[-max_len:]
    for i, c in enumerate(seq[::-1]):
        idx = char_to_id.get(c, 0)
        arr[max_len - 1 - i, idx] = 1.0
    return arr.flatten()

def pad_context(words, ctx_len=10):
    return [""] * max(0, ctx_len - len(words)) + words[-ctx_len:]

def vectorize_context(context_words, w2v_model, ctx_len=10, embed_dim=300):
    vecs = []
    for word in context_words[-ctx_len:]:
        if word in w2v_model:
            vecs.append(w2v_model[word])
        else:
            vecs.append(np.zeros(embed_dim))
    while len(vecs) < ctx_len:
        vecs.insert(0, np.zeros(embed_dim))
    return np.concatenate(vecs, axis=0)

# ---- Lazy Dataset ----

class CharGenLazyDataset(Dataset):
    def __init__(self, json_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50):
        self.json_path = json_path
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len

        # Build byte offsets for all lines
        self.offsets = []
        with open(json_path, encoding="utf-8") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line.encode("utf-8"))

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

# ---- Model ----

class ResNetFFN(nn.Module):
    def __init__(self, context_dim, word_onehot_dim, gen_onehot_dim, char_vocab_size, hidden_dim=600, num_layers=20):
        super().__init__()
        input_dim = context_dim + word_onehot_dim + gen_onehot_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                # Replace ReLU with birely activation function
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, context_vec, misspelled_oh, prefix_oh):
        x = torch.cat([context_vec, misspelled_oh, prefix_oh], dim=1)
        x = self.input_proj(x)
        for layer in self.layers:
            # Apply the custom birely activation instead of the ReLU in the layer
            layer_output = layer[0](x)  # Linear
            layer_output = layer[1](layer_output)  # LayerNorm
            layer_output = birely(layer_output)  # birely instead of ReLU
            x = x + layer_output
        logits = self.output_layer(x)
        return logits

# ---- Training & Prediction ----

def train_model(model, dataloader, vocab_size, epochs=3, save_path="char_autocorrect.pt", 
              local_rank=0, is_distributed=False):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Track best model (only save on rank 0 if distributed)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Set epoch for distributed sampler
        if is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        # Use tqdm only on main process if distributed
        if not is_distributed or local_rank == 0:
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        else:
            loop = dataloader
            
        for context_vec, misspelled_oh, prefix_oh, next_id in loop:
            context_vec = context_vec.to(device)
            misspelled_oh = misspelled_oh.to(device)
            prefix_oh = prefix_oh.to(device)
            next_id = next_id.to(device)
            
            logits = model(context_vec, misspelled_oh, prefix_oh)
            loss = F.cross_entropy(logits, next_id)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if not is_distributed or local_rank == 0:
                if isinstance(loop, tqdm):
                    loop.set_postfix(loss=loss.item())
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count
        
        # Print status and save model (only on main process if distributed)
        if not is_distributed or local_rank == 0:
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            
            # Save model if it's the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                if is_distributed:
                    # Save the module without DDP wrapper
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"New best model saved with loss: {best_loss:.4f}")
            
        # Synchronize processes if distributed
        if is_distributed:
            dist.barrier()

def predict_word(model, w2v_model, char_to_id, id_to_char, context_words, misspelled_word,
                 max_word_len=30, max_gen_len=50, ctx_len=10, max_output_len=30, beam_width=10):
    model.eval()
    with torch.no_grad():
        # Initialize beam with empty sequence
        beams = [(0.0, "")]  # (log_prob, sequence)
        
        for i in range(max_output_len):
            candidates = []
            
            for log_prob, sequence in beams:
                # If sequence is complete, keep it as is
                if sequence.endswith("<eow>") or len(sequence) >= max_output_len:
                    candidates.append((log_prob, sequence))
                    continue
                
                # Get predictions for this sequence
                context_vec = torch.tensor([vectorize_context(context_words, w2v_model, ctx_len)], dtype=torch.float32).to(device)
                misspelled_oh = torch.tensor([one_hot_chars(misspelled_word, char_to_id, max_word_len)], dtype=torch.float32).to(device)
                prefix_oh = torch.tensor([one_hot_chars(sequence, char_to_id, max_gen_len)], dtype=torch.float32).to(device)
                
                logits = model(context_vec, misspelled_oh, prefix_oh)
                log_probs = F.log_softmax(logits, dim=1).squeeze()
                
                # Get top beam_width candidates
                top_k = torch.topk(log_probs, beam_width)
                
                for j in range(beam_width):
                    char_id = top_k.indices[j].item()
                    char_log_prob = top_k.values[j].item()
                    pred_char = id_to_char[char_id]
                    
                    new_sequence = sequence + pred_char
                    new_log_prob = log_prob + char_log_prob
                    
                    candidates.append((new_log_prob, new_sequence))
            
            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Check if all beams are complete
            if all(seq.endswith("<eow>") for _, seq in beams):
                break
        
        # Return top 10 predictions, removing <eow> marker
        results = []
        for log_prob, sequence in beams:
            word = sequence.replace("<eow>", "")
            results.append((word, log_prob))
        
        return results[:10]

def model_matches(model, state_dict):
    """Check if the loaded state_dict fits the model structure. Print all mismatches."""
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    loaded_keys = set(state_dict.keys())

    mismatch = False

    missing = model_keys - loaded_keys
    extra = loaded_keys - model_keys
    if missing:
        print(f"Parameters missing from checkpoint: {missing}")
        mismatch = True
    if extra:
        print(f"Extra parameters in checkpoint: {extra}")
        mismatch = True

    # Check shapes for matching keys
    for k in model_keys & loaded_keys:
        if model_state[k].shape != state_dict[k].shape:
            print(f"Shape mismatch for parameter '{k}': expected {model_state[k].shape}, found {state_dict[k].shape}")
            mismatch = True

    return not mismatch

# ---- CLI ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data", type=str, default="autogen_char_data.json")
    parser.add_argument("--word2vec", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="char_autocorrect.pt")
    # Add multi-GPU arguments
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs with DataParallel")
    parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU to use (if not using multi_gpu)")
    args = parser.parse_args()

    # Setup for distributed training if enabled
    is_distributed = args.distributed
    local_rank = args.local_rank
    
    if is_distributed:
        # Initialize distributed process group
        if local_rank != -1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            print("Error: --distributed requires --local_rank to be set")
            exit(1)
    else:
        # Setup device based on arguments
        device, num_gpus = setup_device(args.gpu, args.multi_gpu)

    from gensim.models import KeyedVectors
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    char_to_id, id_to_char = create_charmap()
    char_vocab_size = len(char_to_id)

    context_dim = args.ctx_len * w2v_model.vector_size
    word_onehot_dim = args.max_word_len * char_vocab_size
    gen_onehot_dim = args.max_gen_len * char_vocab_size

    # Create the base model
    model = ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)

    # Wrap model for multi-GPU training
    if args.multi_gpu and not is_distributed:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")
    elif is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Model wrapped with DistributedDataParallel for GPU {local_rank}")

    if args.train:
        from pathlib import Path
        model_path = args.model
        model_exists = Path(model_path).exists()
        should_resume = False

        if model_exists:
            print(f"Model file '{model_path}' found, checking compatibility...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # If saved as state_dict only
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                
                # For multi-GPU models, we need to handle the state dict differently
                if args.multi_gpu and not is_distributed:
                    # If loading a non-DataParallel model into DataParallel model
                    if not any(k.startswith('module.') for k in checkpoint.keys()):
                        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
                elif is_distributed:
                    # For DDP, similar approach
                    if not any(k.startswith('module.') for k in checkpoint.keys()):
                        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
                else:
                    # If loading a DataParallel model into a non-DataParallel model
                    if any(k.startswith('module.') for k in checkpoint.keys()):
                        checkpoint = {k.replace('module.', ''): v for k in checkpoint.keys()}
                
                if model_matches(model, checkpoint):
                    print("Model structure matches. Resuming training from saved model.")
                    model.load_state_dict(checkpoint)
                    should_resume = True
                else:
                    print("Saved model structure does not match the current code. Exiting for safety.")
                    exit(1)
            except Exception as e:
                print(f"Error loading model: {e}\nExiting.")
                exit(1)
        else:
            print("No existing model found. Training new model from scratch.")

        # Create dataset
        dataset = CharGenLazyDataset(args.data, w2v_model, char_to_id,
                                ctx_len=args.ctx_len, max_word_len=args.max_word_len, max_gen_len=args.max_gen_len)
        
        # Setup data loading based on distributed mode
        if is_distributed:
            # Use DistributedSampler for distributed training
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
            print(f"Rank {local_rank}: Distributed DataLoader created with {len(dataset)} samples")
        else:
            # Regular DataLoader for single GPU or DataParallel
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True
            )
            print(f"DataLoader created with {len(dataset)} samples")
        
        # Train the model
        train_model(
            model, 
            dataloader, 
            vocab_size=char_vocab_size, 
            epochs=args.epochs, 
            save_path=args.model,
            local_rank=local_rank if is_distributed else 0,
            is_distributed=is_distributed
        )
    elif args.predict:
        # For prediction, we need to unwrap the model if it's wrapped in DataParallel or DDP
        if isinstance(model, (nn.DataParallel, DDP)):
            model = model.module
        
        # Load the model weights
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        print("Interactive prediction mode. Press Ctrl+C to exit.")
        try:
            while True:
                print("\nEnter context words (space-separated):")
                context = input("> ").strip().split()
                print("Enter misspelled word:")
                misspelled = input("> ").strip()
                
                # Start timing after input is entered
                start_time = time.time()
                predictions = predict_word(model, w2v_model, char_to_id, id_to_char, context, misspelled,
                                           max_word_len=args.max_word_len, max_gen_len=args.max_gen_len, ctx_len=args.ctx_len)
                end_time = time.time()
                
                prediction_time = end_time - start_time
                print(f"\nTop 10 predictions (computed in {prediction_time:.3f} seconds):")
                for i, (word, log_prob) in enumerate(predictions, 1):
                    print(f"{i:2d}. {word} (log_prob: {log_prob:.3f})")
        except KeyboardInterrupt:
            print("\nExiting...")
