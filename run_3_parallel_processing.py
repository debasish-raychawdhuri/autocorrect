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
import sys
import psutil
from functools import lru_cache
import threading
import queue
import concurrent.futures
from collections import deque

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
    arr = np.zeros((max_len, len(char_to_id)), dtype=np.float32)
    seq = seq[-max_len:]
    for i, c in enumerate(seq[::-1]):
        idx = char_to_id.get(c, 0)
        arr[max_len - 1 - i, idx] = 1.0
    return arr.flatten()

def pad_context(words, ctx_len=10):
    return [""] * max(0, ctx_len - len(words)) + words[-ctx_len:]

@lru_cache(maxsize=100000)
def get_word_vector(word, w2v_model, embed_dim=300):
    """Cached word vector lookup to avoid repeated computation"""
    if word in w2v_model:
        return w2v_model[word]
    return np.zeros(embed_dim)

def vectorize_context(context_words, w2v_model, ctx_len=10, embed_dim=300):
    vecs = []
    for word in context_words[-ctx_len:]:
        vecs.append(get_word_vector(word, w2v_model, embed_dim))
    
    while len(vecs) < ctx_len:
        vecs.insert(0, np.zeros(embed_dim))
    
    return np.concatenate(vecs, axis=0)

# ---- Parallel Batch Processing ----

class ParallelBatchProcessor:
    """Processes batches in parallel using a thread pool"""
    def __init__(self, dataset, batch_size, num_workers, max_queue_size=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.stop_event = threading.Event()
        self.worker_threads = []
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)
        self.index_position = 0
        self.epoch_complete = threading.Event()
        
        # Start worker threads
        for _ in range(num_workers):
            thread = threading.Thread(target=self._worker_loop)
            thread.daemon = True
            self.worker_threads.append(thread)
            thread.start()
    
    def _worker_loop(self):
        """Worker thread that processes batches"""
        while not self.stop_event.is_set():
            # Get a batch of indices to process
            batch_indices = self._get_next_batch_indices()
            if batch_indices is None:
                # No more indices for this epoch
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue
                
            # Process the batch
            batch_data = self._process_batch(batch_indices)
            
            # Put the processed batch in the queue
            try:
                self.queue.put(batch_data, block=True, timeout=5.0)
            except queue.Full:
                if self.stop_event.is_set():
                    break
                # If queue is full, wait and try again
                time.sleep(0.1)
    
    def _get_next_batch_indices(self):
        """Get the next batch of indices to process"""
        with threading.Lock():
            if self.index_position >= len(self.indices):
                return None
            
            end_pos = min(self.index_position + self.batch_size, len(self.indices))
            batch_indices = self.indices[self.index_position:end_pos]
            self.index_position += len(batch_indices)
            
            # Check if we've completed the epoch
            if self.index_position >= len(self.indices):
                self.epoch_complete.set()
            
            return batch_indices
    
    def _process_batch(self, indices):
        """Process a batch of indices into tensors"""
        context_vecs = []
        misspelled_ohs = []
        prefix_ohs = []
        next_ids = []
        
        for idx in indices:
            context_vec, misspelled_oh, prefix_oh, next_id = self.dataset[idx]
            context_vecs.append(context_vec)
            misspelled_ohs.append(misspelled_oh)
            prefix_ohs.append(prefix_oh)
            next_ids.append(next_id)
        
        # Stack into batch tensors
        return (
            torch.stack(context_vecs),
            torch.stack(misspelled_ohs),
            torch.stack(prefix_ohs),
            torch.stack(next_ids)
        )
    
    def get_batch(self, timeout=10.0):
        """Get a processed batch from the queue"""
        try:
            return self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            if self.epoch_complete.is_set() and self.queue.empty():
                # Epoch is complete and queue is empty
                return None
            # Queue is empty but epoch is not complete
            return None
    
    def reset_for_new_epoch(self):
        """Reset for a new epoch"""
        with threading.Lock():
            self.index_position = 0
            random.shuffle(self.indices)
            self.epoch_complete.clear()
            
            # Clear the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
    
    def shutdown(self):
        """Shutdown the processor"""
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        for thread in self.worker_threads:
            thread.join(timeout=2.0)

# ---- Memory-Optimized Dataset ----

class CharGenMemoryDataset(Dataset):
    def __init__(self, data_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50):
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len
        
        print(f"🔄 Loading optimized dataset from {data_path}...")
        start_time = time.time()
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Load the optimized data
        data = torch.load(data_path)
        self.contexts = data['contexts']
        self.misspelled = data['misspelled']
        self.targets = data['targets']
        self.sample_offsets = data['sample_offsets']
        self.total_samples = data['total_samples']
        
        # Report memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage after loading: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB)")
        
        load_time = time.time() - start_time
        print(f"✅ Loaded {len(self.contexts):,} samples with {self.total_samples:,} training examples in {load_time:.2f} seconds")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search to find which original sample this idx belongs to
        left, right = 0, len(self.sample_offsets) - 2  # -2 because we're checking idx < self.sample_offsets[mid+1]
        while left <= right:
            mid = (left + right) // 2
            if self.sample_offsets[mid] <= idx < self.sample_offsets[mid + 1]:
                sample_idx = mid
                break
            elif idx < self.sample_offsets[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            sample_idx = 0  # Fallback
        
        # Get the sample
        context = self.contexts[sample_idx]
        misspelled = self.misspelled[sample_idx]
        target = self.targets[sample_idx]
        
        # Calculate which prefix to use
        prefix_idx = idx - self.sample_offsets[sample_idx]
        
        # Generate the specific prefix for this index
        if prefix_idx < len(target):
            prefix = target[:prefix_idx]
            next_char = target[prefix_idx]
        else:
            prefix = target
            next_char = "<eow>"
        
        # Process the data
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

def train_model_parallel(model, dataset, batch_size, num_workers, vocab_size, epochs=3, save_path="char_autocorrect.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Fix for FutureWarning about GradScaler
    try:
        # New API (PyTorch 2.0+)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    except TypeError:
        # Fallback to old API
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Track metrics for early stopping and learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    # Create parallel batch processor
    print(f"Starting parallel batch processor with {num_workers} workers")
    batch_processor = ParallelBatchProcessor(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        max_queue_size=num_workers * 2  # Allow for 2 batches per worker in queue
    )
    
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            start_time = time.time()
            
            # Reset batch processor for new epoch
            batch_processor.reset_for_new_epoch()
            
            # Calculate total batches for progress bar
            total_batches = (len(dataset) + batch_size - 1) // batch_size
            
            # Create progress bar
            loop = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            while True:
                # Get next batch from parallel processor
                batch = batch_processor.get_batch(timeout=10.0)
                
                # Check if epoch is complete
                if batch is None:
                    if batch_processor.epoch_complete.is_set():
                        break
                    # Wait a bit and try again
                    time.sleep(0.1)
                    continue
                
                # Unpack batch
                context_vec, misspelled_oh, prefix_oh, next_id = batch
                
                # Move data to device
                context_vec = context_vec.to(device, non_blocking=True)
                misspelled_oh = misspelled_oh.to(device, non_blocking=True)
                prefix_oh = prefix_oh.to(device, non_blocking=True)
                next_id = next_id.to(device, non_blocking=True)
                
                # Mixed precision training if available
                if scaler is not None:
                    try:
                        # Try new API first
                        with torch.amp.autocast('cuda'):
                            logits = model(context_vec, misspelled_oh, prefix_oh)
                            loss = F.cross_entropy(logits, next_id)
                    except (AttributeError, TypeError):
                        # Fall back to old API
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
                
                # Update progress bar
                loop.update(1)
                loop.set_postfix(
                    loss=f"{batch_loss:.4f}", 
                    avg=f"{epoch_loss/batch_count:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    time=f"{(time.time()-start_time)/60:.1f}m"
                )
            
            # End of epoch
            loop.close()
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
    
    finally:
        # Always shut down the batch processor
        print("Shutting down parallel batch processor...")
        batch_processor.shutdown()

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
    # Try to install psutil if not available
    try:
        import psutil
    except ImportError:
        print("psutil not found, trying to install...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
            print("psutil installed successfully")
        except Exception as e:
            print(f"Could not install psutil: {e}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data", type=str, required=True, help="Path to optimized data file")
    parser.add_argument("--word2vec", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)  # Increased default batch size for 128GB RAM
    parser.add_argument("--model", type=str, default="char_autocorrect.pt")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker threads for parallel processing")
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

        # Print system memory information
        try:
            mem_info = psutil.virtual_memory()
            print(f"System memory: {mem_info.total / (1024**3):.1f} GB total, "
                  f"{mem_info.available / (1024**3):.1f} GB available")
            
            # Print CPU information
            cpu_count = os.cpu_count()
            print(f"CPU cores: {cpu_count}")
            
            # Adjust number of workers based on available cores
            if args.num_workers > cpu_count:
                suggested_workers = max(1, cpu_count - 2)  # Leave 2 cores for system
                print(f"Warning: Requested {args.num_workers} workers but only {cpu_count} CPU cores available.")
                print(f"Suggestion: Consider using --num_workers={suggested_workers}")
        except:
            print("Could not get system information")

        # Now proceed to dataset and training with memory-optimized dataset
        print(f"Creating memory-optimized dataset from {args.data}")
        dataset = CharGenMemoryDataset(
            args.data, w2v_model, char_to_id,
            ctx_len=args.ctx_len, 
            max_word_len=args.max_word_len, 
            max_gen_len=args.max_gen_len
        )
        
        # Use parallel training
        print(f"Starting parallel training with {args.num_workers} worker threads")
        train_model_parallel(
            model=model, 
            dataset=dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vocab_size=char_vocab_size, 
            epochs=args.epochs, 
            save_path=args.model
        )
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
