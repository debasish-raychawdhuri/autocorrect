import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import string
import os
import time
from batch_sampler import BatchSamplerByChunks
from functools import lru_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---- Char Map ----

def create_charmap():
    printable = string.printable.replace('"', '')  # Avoid JSON quote issues
    char_list = list(printable) + ["<eow>"]
    char_to_id = {c: i for i, c in enumerate(char_list)}
    id_to_char = {i: c for i, c in enumerate(char_list)}
    return char_to_id, id_to_char

# ---- Input Preparation ----

def one_hot_chars(seq, char_to_id, max_len):
    """Optimized one-hot encoding using numpy operations"""
    char_vocab_size = len(char_to_id)
    arr = np.zeros((max_len, char_vocab_size), dtype=np.float32)
    seq = seq[-max_len:]  # Truncate if too long
    
    # Process in reverse order to match original behavior
    for i, c in enumerate(seq[::-1]):
        idx = char_to_id.get(c, 0)
        arr[max_len - 1 - i, idx] = 1.0
    
    return arr.reshape(-1)  # Flatten

@lru_cache(maxsize=100000)
def get_word_vector(word, w2v_model, embed_dim=300):
    """Cached word vector lookup to avoid repeated computation"""
    if word in w2v_model:
        return w2v_model[word]
    return np.zeros(embed_dim)

def vectorize_context(context_words, w2v_model, ctx_len=10, embed_dim=300):
    """Vectorize context with cached word vectors"""
    vecs = []
    for word in context_words[-ctx_len:]:
        vecs.append(get_word_vector(word, w2v_model, embed_dim))
    
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

        # Check if cached index exists
        index_path = f"{json_path}.index"
        
        if os.path.exists(index_path):
            print(f"📂 Loading cached index from {index_path}")
            try:
                index_data = torch.load(index_path)
                self.offsets = index_data['offsets']
                self.sample_lengths = index_data['sample_lengths']
                self.cumulative_lengths = index_data['cumulative_lengths']
                self.total_samples = index_data['total_samples']
                
                print(f"✅ Loaded index: {self.total_samples:,} samples from {len(self.offsets):,} original samples")
                return
            except Exception as e:
                print(f"⚠️ Failed to load cached index: {e}")
                print("Building new index...")

        # Build byte offsets for all lines (original samples)
        self.offsets = []
        self.sample_lengths = []  # Track how many prefix samples each original sample generates
        
        # Count total lines first for progress bar
        print("Counting lines in dataset...")
        with open(json_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print("Building sample index...")
        with open(json_path, encoding="utf-8") as f:
            pos = 0
            total_expanded_samples = 0
            
            # Create progress bar for indexing
            pbar = tqdm(total=total_lines, desc="Indexing samples", unit="samples")
            
            for line_idx, line in enumerate(f):
                self.offsets.append(pos)
                pos += len(line.encode("utf-8"))
                
                # Count how many prefix samples this line will generate
                sample = json.loads(line.strip())
                target = sample["target"]
                prefix_count = len(target) + 1  # +1 for the <eow> case
                self.sample_lengths.append(prefix_count)
                total_expanded_samples += prefix_count
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(expanded_samples=f"{total_expanded_samples:,}")
            
            pbar.close()
        
        # Build cumulative index to map global sample index to (line_idx, prefix_idx)
        print("Building cumulative index...")
        self.cumulative_lengths = []
        cumsum = 0
        for length in tqdm(self.sample_lengths, desc="Building index", unit="samples"):
            cumsum += length
            self.cumulative_lengths.append(cumsum)
        
        self.total_samples = total_expanded_samples
        
        # Save the index for future use
        print(f"💾 Saving index to {index_path}")
        try:
            torch.save({
                'offsets': self.offsets,
                'sample_lengths': self.sample_lengths,
                'cumulative_lengths': self.cumulative_lengths,
                'total_samples': self.total_samples
            }, index_path)
            print(f"✅ Index saved successfully")
        except Exception as e:
            print(f"⚠️ Failed to save index: {e}")
        
        print(f"✅ Dataset ready: {self.total_samples:,} samples from {len(self.offsets):,} original samples")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search to find which original sample and which prefix this idx corresponds to
        left, right = 0, len(self.cumulative_lengths) - 1
        while left <= right:
            mid = (left + right) // 2
            if mid > 0 and self.cumulative_lengths[mid-1] <= idx < self.cumulative_lengths[mid]:
                line_idx = mid
                break
            elif idx < self.cumulative_lengths[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            line_idx = 0  # Fallback
        
        # Calculate the prefix index within this sample
        prefix_idx = idx - (self.cumulative_lengths[line_idx - 1] if line_idx > 0 else 0)
        
        # Load the original sample - use binary mode for faster I/O
        with open(self.json_path, 'rb') as f:
            f.seek(self.offsets[line_idx])
            line = f.readline().decode('utf-8')
            sample = json.loads(line.strip())
        
        # Parse input text to get context and misspelled word
        input_text = sample["input"].split()
        misspelled = input_text[-1]  # Last word is misspelled
        context = input_text[:-1]    # Rest is context
        target = sample["target"]    # Correct word
        
        # Generate the specific prefix for this index
        if prefix_idx < len(target):
            prefix = target[:prefix_idx]
            next_char = target[prefix_idx]
        else:
            prefix = target
            next_char = "<eow>"
        
        # Process as before - use optimized functions
        context = pad_context(context, self.ctx_len)
        context_vec = vectorize_context(context, self.w2v_model, self.ctx_len)
        misspelled_oh = one_hot_chars(misspelled, self.char_to_id, self.max_word_len)
        prefix_oh = one_hot_chars(prefix, self.char_to_id, self.max_gen_len)
        
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

# ---- Custom Activation Function ----

class BiReLU(nn.Module):
    """BiReLU activation function: birelu(x) = relu(x) - 0.3 * relu(-x)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu(x) - 0.3 * F.relu(-x)

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
                BiReLU()
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, context_vec, misspelled_oh, prefix_oh):
        x = torch.cat([context_vec, misspelled_oh, prefix_oh], dim=1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        logits = self.output_layer(x)
        return logits

# ---- Training & Prediction ----

def train_model(model, dataloader, vocab_size, epochs=3, save_path="char_autocorrect.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Track metrics for early stopping and learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for context_vec, misspelled_oh, prefix_oh, next_id in loop:
            # Move data to device
            context_vec = context_vec.to(device, non_blocking=True)
            misspelled_oh = misspelled_oh.to(device, non_blocking=True)
            prefix_oh = prefix_oh.to(device, non_blocking=True)
            next_id = next_id.to(device, non_blocking=True)
            
            # Mixed precision training if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(context_vec, misspelled_oh, prefix_oh)
                    loss = F.cross_entropy(logits, next_id)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                logits = model(context_vec, misspelled_oh, prefix_oh)
                loss = F.cross_entropy(logits, next_id)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_count += 1
            
            # Update progress bar with more info
            loop.set_postfix(
                loss=f"{batch_loss:.4f}", 
                avg=f"{epoch_loss/batch_count:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                time=f"{(time.time()-start_time)/60:.1f}m"
            )
        
        # End of epoch
        avg_loss = epoch_loss / batch_count
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.4f}, Time: {epoch_time/60:.1f} minutes")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_loss)
        
        # Save model at end of each epoch
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{save_path}.best")
            print(f"📈 New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⚠️ Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Force garbage collection between epochs
        torch.cuda.empty_cache()

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
    parser.add_argument("--batch_size", type=int, default=512)  # Increased default batch size
    parser.add_argument("--model", type=str, default="char_autocorrect.pt")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of samples to process in each chunk")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    args = parser.parse_args()

    from gensim.models import KeyedVectors
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    char_to_id, id_to_char = create_charmap()
    char_vocab_size = len(char_to_id)

    context_dim = args.ctx_len * w2v_model.vector_size
    word_onehot_dim = args.max_word_len * char_vocab_size
    gen_onehot_dim = args.max_gen_len * char_vocab_size

    model = ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)

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

        # Now proceed to dataset and training as before
        dataset = CharGenLazyDataset(args.data, w2v_model, char_to_id,
                                ctx_len=args.ctx_len, max_word_len=args.max_word_len, max_gen_len=args.max_gen_len)
        
        # Use custom batch sampler to process data in chunks
        batch_sampler = BatchSamplerByChunks(
            dataset_size=len(dataset),
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            shuffle=True
        )
        
        # Use more workers to keep CPU busy and GPU fed
        dataloader = DataLoader(
            dataset, 
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        
        train_model(model, dataloader, vocab_size=char_vocab_size, epochs=args.epochs, save_path=args.model)   
    elif args.predict:
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

