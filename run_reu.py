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
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
import string
import os
import time
import signal
import torch.onnx
import yaml
from datetime import timedelta
from multiprocessing import shared_memory
import pickle

# Global shared memory for word2vec
_shared_w2v_memory = None
_shared_w2v_metadata = None

class SharedWord2Vec:
    """Wrapper for word2vec model in shared memory"""
    
    def __init__(self, shm_name, metadata):
        self.shm_name = shm_name
        self.metadata = metadata
        self.vector_size = metadata['vector_size']
        self.vocab_size = metadata['vocab_size']
        self.word_to_index = metadata['word_to_index']
        
        # Attach to existing shared memory
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.vectors = np.ndarray(
            (self.vocab_size, self.vector_size), 
            dtype=np.float32, 
            buffer=self.shm.buf
        )
    
    def __contains__(self, word):
        return word in self.word_to_index
    
    def __getitem__(self, word):
        if word not in self.word_to_index:
            raise KeyError(f"Word '{word}' not in vocabulary")
        idx = self.word_to_index[word]
        return self.vectors[idx]

def create_shared_word2vec(w2v_model):
    """Create shared memory version of word2vec model"""
    global _shared_w2v_memory, _shared_w2v_metadata
    
    print("Creating shared memory for word2vec model...")
    
    # Extract words and vectors
    words = list(w2v_model.word_vectors.keys())
    vectors = np.array([w2v_model.word_vectors[word] for word in words], dtype=np.float32)
    
    vocab_size, vector_size = vectors.shape
    
    # Create shared memory
    shm_size = vectors.nbytes
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    
    # Copy vectors to shared memory
    shared_vectors = np.ndarray(vectors.shape, dtype=np.float32, buffer=shm.buf)
    shared_vectors[:] = vectors[:]
    
    # Create metadata
    metadata = {
        'vector_size': vector_size,
        'vocab_size': vocab_size,
        'word_to_index': {word: i for i, word in enumerate(words)}
    }
    
    _shared_w2v_memory = shm
    _shared_w2v_metadata = metadata
    
    print(f"Created shared memory: {shm_size / 1024 / 1024:.1f} MB for {vocab_size} words")
    
    return SharedWord2Vec(shm.name, metadata)

def get_shared_word2vec():
    """Get shared word2vec instance for worker processes"""
    global _shared_w2v_metadata
    if _shared_w2v_metadata is None:
        raise RuntimeError("Shared word2vec not initialized")
    
    # Get shared memory name from environment or global
    shm_name = os.environ.get('SHARED_W2V_NAME')
    if shm_name is None:
        raise RuntimeError("Shared word2vec name not found")
    
    return SharedWord2Vec(shm_name, _shared_w2v_metadata)

def cleanup_shared_word2vec():
    """Cleanup shared memory"""
    global _shared_w2v_memory
    if _shared_w2v_memory is not None:
        _shared_w2v_memory.close()
        _shared_w2v_memory.unlink()
        _shared_w2v_memory = None
        print("Shared word2vec memory cleaned up")

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

# Check for distributed environment variables
def get_distributed_info():
    """Get distributed training information from environment variables"""
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    # If any of these are not set, check for SLURM variables
    if rank == -1 or world_size == -1 or local_rank == -1:
        if 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
    
    return rank, world_size, local_rank

# Default device setup (will be updated in main)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Initial device setup: {device}")

# Custom birelu activation function: birelu(x) = relu(x) - 0.3 * relu(-x)
def birelu(x):
    return F.relu(x) - 0.3 * F.relu(-x)

class BiReLU(nn.Module):
    """BiReLU activation module for use in nn.Sequential"""
    def forward(self, x):
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

class CharGenStreamingDataset(IterableDataset):
    def __init__(self, data_dir, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50, num_workers=4, use_shared_memory=False):
        # Handle both single file and directory input
        if os.path.isfile(data_dir):
            # Single file mode (backward compatibility)
            self.file_paths = [data_dir]
            self.use_worker_files = False
        else:
            # Multiple files mode - each worker gets its own file
            self.file_paths = []
            for filename in sorted(os.listdir(data_dir)):
                if filename.endswith('.json') and filename != 'metadata.json':
                    self.file_paths.append(os.path.join(data_dir, filename))
            self.use_worker_files = True
        
        print(f"Found {len(self.file_paths)} data files, worker-specific files: {self.use_worker_files}")
        
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len
        self.num_workers = num_workers
        self.use_shared_memory = use_shared_memory
        
        # Store shared memory metadata for workers
        if use_shared_memory and hasattr(w2v_model, 'shm_name'):
            self.shm_name = w2v_model.shm_name
            self.w2v_metadata = w2v_model.metadata
        else:
            self.shm_name = None
            self.w2v_metadata = None

        # For streaming mode, read exact sample counts from metadata
        if self.use_worker_files:
            print(f"Using streaming mode with {len(self.file_paths)} worker files...")
            
            # Try to read metadata file for exact counts
            data_dir = os.path.dirname(self.file_paths[0]) if self.file_paths else "."
            metadata_path = os.path.join(data_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.total_samples = metadata["total_samples"]
                self.file_metadata = {f["filename"]: f["samples"] for f in metadata["files"]}
                print(f"Loaded metadata: {self.total_samples:,} total samples across {len(self.file_paths)} files")
            else:
                # Fallback to quick estimation if no metadata
                print("No metadata.json found, using quick estimation...")
                self.total_samples = 0
                for file_path in self.file_paths:
                    with open(file_path, encoding="utf-8") as f:
                        sample_lines = 0
                        for i, line in enumerate(f):
                            sample_lines += 1
                            if i >= 1000:
                                break
                        
                        f.seek(0, 2)
                        file_size = f.tell()
                        if sample_lines > 0:
                            avg_line_size = f.tell() / sample_lines if i >= 1000 else file_size / sample_lines
                            estimated_lines = int(file_size / avg_line_size) if avg_line_size > 0 else sample_lines
                            self.total_samples += estimated_lines
                
                print(f"Estimated total samples: {self.total_samples:,} (no metadata available)")
        else:
            # Single file mode still uses indexing for compatibility
            print(f"Building dataset index with {num_workers} workers...")
            start_time = time.time()
            
            if num_workers > 1:
                self.offsets = self._build_offsets_parallel(num_workers)
            else:
                self.offsets = self._build_offsets_sequential()
                
            end_time = time.time()
            print(f"Dataset index built with {len(self.offsets)} samples in {end_time - start_time:.2f} seconds")

    def _build_offsets_for_file(self, file_path):
        """Build offsets for a single file"""
        offsets = []
        with open(file_path, encoding="utf-8") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line.encode("utf-8"))
        return offsets

    def _build_offsets_sequential(self):
        """Build offsets sequentially"""
        offsets = []
        with open(self.file_paths[0], encoding="utf-8") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line.encode("utf-8"))
        return offsets
    
    def _build_offsets_parallel(self, num_workers):
        """Build offsets using multiple processes"""
        # Fall back to sequential processing for now to avoid pickling issues
        return self._build_offsets_sequential()

    def __iter__(self):
        # Get worker-specific word2vec model
        if self.use_shared_memory and self.shm_name:
            # Worker process - attach to shared memory
            w2v_model = SharedWord2Vec(self.shm_name, self.w2v_metadata)
        else:
            # Use the original model
            w2v_model = self.w2v_model
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Each worker processes multiple files assigned to it
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Assign files to workers in round-robin fashion
            worker_files = [self.file_paths[i] for i in range(len(self.file_paths)) if i % num_workers == worker_id]
        else:
            # Single process mode - use all files
            worker_files = self.file_paths
        
        # Process all files assigned to this worker
        for file_path in worker_files:
            # Open file ONCE and read through it exactly once per epoch
            with open(file_path, encoding="utf-8", buffering=8*1024*1024) as f:  # 8MB buffer
                for line in f:
                    sample = json.loads(line.strip())
                    context = pad_context(sample["context"], self.ctx_len)
                    misspelled = sample["misspelled"]
                    prefix = sample["generated_prefix"]
                    next_char = sample["next_char"]
                    context_vec = vectorize_context(context, w2v_model, self.ctx_len)
                    misspelled_oh = one_hot_chars(misspelled, self.char_to_id, self.max_word_len)
                    prefix_oh = one_hot_chars(prefix, self.char_to_id, self.max_gen_len)
                    
                    if next_char == "<eow>":
                        next_id = self.char_to_id["<eow>"]
                    else:
                        next_id = self.char_to_id.get(next_char, 0)
                    
                    yield (
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
                BiReLU()
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, context_vec, misspelled_oh, prefix_oh):
        x = torch.cat([context_vec, misspelled_oh, prefix_oh], dim=1)
        x = self.input_proj(x)
        for layer in self.layers:
            # ResNet connection with BiReLU activation
            layer_output = layer(x)  # Linear -> LayerNorm -> BiReLU
            x = x + layer_output
        logits = self.output_layer(x)
        return logits

# ---- Model Format Detection ----

def get_model_format(file_path):
    """Determine model format based on file extension"""
    if file_path.endswith('.pt') or file_path.endswith('.pth'):
        return 'pytorch'
    elif file_path.endswith('.onnx'):
        return 'onnx'
    else:
        # Default to ONNX if no recognized extension
        return 'onnx'

def save_model(model, file_path, context_dim=None, word_onehot_dim=None, gen_onehot_dim=None, is_distributed=False):
    """Save model in the format specified by file extension"""
    format_type = get_model_format(file_path)
    
    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_distributed else model
    
    if format_type == 'pytorch':
        torch.save(actual_model.state_dict(), file_path)
    elif format_type == 'onnx':
        if context_dim is None or word_onehot_dim is None or gen_onehot_dim is None:
            raise ValueError("ONNX export requires context_dim, word_onehot_dim, and gen_onehot_dim")
        save_model_to_onnx(actual_model, file_path, 
                          (1, context_dim), (1, word_onehot_dim), (1, gen_onehot_dim))

# ---- ONNX Export ----

def save_model_to_onnx(model, onnx_path, context_shape, misspelled_shape, prefix_shape):
    """Save model to ONNX format"""
    model.eval()
    
    # Create dummy inputs with the right shapes
    dummy_context = torch.randn(context_shape).to(next(model.parameters()).device)
    dummy_misspelled = torch.randn(misspelled_shape).to(next(model.parameters()).device)
    dummy_prefix = torch.randn(prefix_shape).to(next(model.parameters()).device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_context, dummy_misspelled, dummy_prefix),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['context_vec', 'misspelled_oh', 'prefix_oh'],
        output_names=['logits'],
        dynamic_axes={
            'context_vec': {0: 'batch_size'},
            'misspelled_oh': {0: 'batch_size'},
            'prefix_oh': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

# ---- Training & Prediction ----

def train_model(model, dataloader, vocab_size, epochs=3, save_path="char_autocorrect.pt", 
              local_rank=0, is_distributed=False, num_workers=4, total_samples=None, batch_size=32,
              context_dim=None, word_onehot_dim=None, gen_onehot_dim=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Track best model (only save on rank 0 if distributed)
    best_loss = float('inf')
    
    # Global flag for save on demand
    save_requested = [False]
    
    def save_on_demand(signum, frame):
        save_requested[0] = True
        print("\n💾 Save requested! Will save model after current batch...")
    
    # Set up Ctrl+S handler (SIGUSR1 is more reliable than trying to capture Ctrl+S)
    # Use: kill -USR1 <pid> to trigger save
    signal.signal(signal.SIGUSR1, save_on_demand)
    
    # Set up data prefetcher for faster data loading
    from torch.utils.data import DataLoader, Dataset
    
    # Initialize CUDA context and synchronize all processes
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if is_distributed else 0)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Synchronize all processes before training
    if is_distributed:
        dist.barrier()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Set epoch for distributed sampler
        if is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        # Use tqdm only on main process if distributed
        if not is_distributed or local_rank == 0:
            # Calculate total batches if we have total_samples
            total_batches = total_samples // batch_size if total_samples else None
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", total=total_batches)
        else:
            loop = dataloader
            
        for context_vec, misspelled_oh, prefix_oh, next_id in loop:
            try:
                # Ensure tensors are on the correct device with proper synchronization
                context_vec = context_vec.to(device, non_blocking=True)
                misspelled_oh = misspelled_oh.to(device, non_blocking=True)
                prefix_oh = prefix_oh.to(device, non_blocking=True)
                next_id = next_id.to(device, non_blocking=True)
                
                # Wait for data transfer to complete before forward pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                logits = model(context_vec, misspelled_oh, prefix_oh)
                loss = F.cross_entropy(logits, next_id)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Synchronize gradients before clipping in distributed mode
                if is_distributed:
                    torch.cuda.synchronize()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                # Add explicit CUDA synchronization for multi-GPU stability
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if not is_distributed or local_rank == 0:
                    if isinstance(loop, tqdm):
                        loop.set_postfix(loss=loss.item())
                
                # Check if save was requested (only save on main process)
                if save_requested[0] and (not is_distributed or local_rank == 0):
                    print(f"\n💾 Saving model on demand...")
                    
                    save_model(model, save_path, context_dim, word_onehot_dim, gen_onehot_dim, is_distributed)
                    
                    print(f"✅ Model saved to {save_path}")
                    save_requested[0] = False
                        
            except RuntimeError as e:
                if "CUDA" in str(e) or "illegal memory access" in str(e):
                    print(f"CUDA error on rank {local_rank}: {e}")
                    # Clear CUDA cache and try to recover
                    torch.cuda.empty_cache()
                    if is_distributed:
                        dist.barrier()
                    continue
                else:
                    raise e
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count
        
        # Print status and save model (only on main process if distributed)
        if not is_distributed or local_rank == 0:
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            
            # Save model if it's the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                save_model(model, save_path, context_dim, word_onehot_dim, gen_onehot_dim, is_distributed)
                print(f"New best model saved with loss: {best_loss:.4f} → {save_path}")
            
        # Synchronize processes if distributed
        if is_distributed:
            # Clear CUDA cache before barrier to prevent memory issues
            torch.cuda.empty_cache()
            dist.barrier()
            
        # Additional cleanup between epochs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_config_args(config, args, provided_args):
    """Merge config file with command line args, giving priority to command line"""
    # Start with config defaults
    merged = config.copy() if config else {}
    
    # Override with command line args (only explicitly provided values)
    boolean_flags = ['train', 'predict', 'multi_gpu', 'distributed']
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key in provided_args:
            merged[key] = value
        elif value is not None and key not in boolean_flags:
            # For non-boolean flags, treat non-None as explicitly provided
            merged[key] = value
    
    return merged

if __name__ == "__main__":
    # Set multiprocessing start method
    import multiprocessing as mp
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data_dir", type=str, help="Directory containing training data files or single JSON file")
    parser.add_argument("--word2vec", type=str, help="Custom word2vec file (.npz, .pkl, .json)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_word_len", type=int, help="Maximum word length")
    parser.add_argument("--max_gen_len", type=int, help="Maximum generation length")
    parser.add_argument("--ctx_len", type=int, help="Context length")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--model", type=str, help="Model file path")
    # Add model architecture arguments
    parser.add_argument("--hidden_dim", type=int, help="Hidden layer width")
    parser.add_argument("--num_layers", type=int, help="Number of hidden layers")
    # Add multi-GPU arguments
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs with DataParallel")
    parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument("--local_rank", "--local-rank", type=int, help="Local rank for distributed training")
    parser.add_argument("--gpu", type=int, help="Specific GPU to use (if not using multi_gpu)")
    # Add CPU utilization arguments
    parser.add_argument("--num_workers", type=int, help="Number of worker processes for data loading")
    parser.add_argument("--mp_start_method", type=str, choices=['fork', 'spawn', 'forkserver'],
                        help="Multiprocessing start method")
    
    cmd_args = parser.parse_args()
    
    # Load config file if specified
    config = {}
    if cmd_args.config:
        config = load_config(cmd_args.config)
        print(f"📄 Loaded config from: {cmd_args.config}")
    
    # Track which boolean args were explicitly provided
    provided_args = set()
    boolean_flags = ['train', 'predict', 'multi_gpu', 'distributed']
    for flag in boolean_flags:
        if f'--{flag}' in sys.argv:
            provided_args.add(flag)
    
    # Merge config with command line args
    merged_config = merge_config_args(config, cmd_args, provided_args)
    
    # Convert back to argparse Namespace with defaults
    class Config:
        def __init__(self, **kwargs):
            # Set defaults
            self.train = kwargs.get('train', False)
            self.predict = kwargs.get('predict', False)
            self.data_dir = kwargs.get('data_dir', 'training_data')
            self.word2vec = kwargs.get('word2vec')
            self.epochs = kwargs.get('epochs', 3)
            self.max_word_len = kwargs.get('max_word_len', 50)
            self.max_gen_len = kwargs.get('max_gen_len', 50)
            self.ctx_len = kwargs.get('ctx_len', 10)
            self.batch_size = kwargs.get('batch_size', 32)
            self.model = kwargs.get('model', 'char_autocorrect.onnx')
            self.hidden_dim = kwargs.get('hidden_dim', 600)
            self.num_layers = kwargs.get('num_layers', 30)
            self.multi_gpu = kwargs.get('multi_gpu', False)
            self.distributed = kwargs.get('distributed', False)
            self.local_rank = kwargs.get('local_rank', -1)
            self.gpu = kwargs.get('gpu')
            self.num_workers = kwargs.get('num_workers')
            self.mp_start_method = kwargs.get('mp_start_method', 'fork')
    
    args = Config(**merged_config)
    
    # Print all configuration options for user confirmation
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {'Training' if args.train else 'Prediction' if args.predict else 'Unknown'}")
    print(f"Data directory: {args.data_dir}")
    print(f"Word2Vec file: {args.word2vec}")
    print(f"Model file: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Context length: {args.ctx_len}")
    print(f"Max word length: {args.max_word_len}")
    print(f"Max generation length: {args.max_gen_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dimensions: {args.hidden_dim}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print(f"Distributed: {args.distributed}")
    print(f"GPU: {args.gpu}")
    if args.num_workers:
        print(f"Data loading workers: {args.num_workers}")
    print(f"Multiprocessing method: {args.mp_start_method}")
    print("=" * 60)
    
    # Get user confirmation
    response = input("Proceed with training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        exit(0)
    print()

    # Validate required args
    if not args.word2vec:
        print("Error: --word2vec is required (or specify in config file)")
        exit(1)

    # Setup for distributed training if enabled
    is_distributed = args.distributed
    local_rank = args.local_rank
    
    # Set multiprocessing start method
    try:
        mp.set_start_method(args.mp_start_method)
        print(f"Using multiprocessing start method: {args.mp_start_method}")
    except RuntimeError:
        print(f"Multiprocessing start method already set to: {mp.get_start_method()}")
    
    # Determine optimal number of workers for data loading
    if args.num_workers is None:
        # Use 80% of available CPUs for data loading
        cpu_count = mp.cpu_count()
        args.num_workers = max(1, int(cpu_count * 0.8))
        print(f"Auto-detected {cpu_count} CPUs, using {args.num_workers} worker processes")
    else:
        print(f"Using {args.num_workers} worker processes as specified")
    
    if is_distributed:
        # Check environment variables first
        env_rank, env_world_size, env_local_rank = get_distributed_info()
        
        # Use environment variables if available, otherwise use command line args
        if env_local_rank != -1:
            local_rank = env_local_rank
            print(f"Using local_rank={local_rank} from environment variables")
        
        # Initialize distributed process group with better error handling
        if local_rank != -1:
            try:
                # Clear CUDA cache before initialization
                torch.cuda.empty_cache()
                
                # Set device before initializing process group
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                
                # Initialize CUDA context
                torch.cuda.init()
                
                # Check if we're using PyTorch's distributed launch
                if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
                    # Using env:// initialization method which uses these environment variables
                    print(f"Initializing process group with local_rank={local_rank}")
                    dist.init_process_group(
                        backend='nccl', 
                        init_method='env://', 
                        timeout=timedelta(seconds=300),
                        device_id=device  # Pass device object, not integer
                    )
                else:
                    # Fallback to default initialization
                    print(f"Initializing process group with default settings, local_rank={local_rank}")
                    dist.init_process_group(
                        backend='nccl', 
                        timeout=timedelta(seconds=300),
                        device_id=device  # Pass device object, not integer
                    )
                
                # Set memory fraction to avoid OOM (do this early)
                torch.cuda.set_per_process_memory_fraction(0.7, device=local_rank)
                
                print(f"Process {local_rank} using device: {device}")
                
                # Test CUDA operations with small tensor
                test_tensor = torch.randn(10, 10).to(device)
                _ = test_tensor.sum()
                torch.cuda.synchronize()
                print(f"CUDA operations test passed on rank {local_rank}")
                
                # Clear test tensor
                del test_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error initializing distributed training on rank {local_rank}: {e}")
                import traceback
                traceback.print_exc()
                # Clean up and exit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                exit(1)
        else:
            print("Error: --distributed requires --local_rank to be set or environment variables")
            exit(1)
    else:
        # Setup device based on arguments
        device, num_gpus = setup_device(args.gpu, args.multi_gpu)

    from custom_word2vec import load_custom_word2vec
    
    # Load word2vec and create shared memory version for workers
    print("Loading word2vec model...")
    original_w2v_model = load_custom_word2vec(args.word2vec)
    
    # Create shared memory version for DataLoader workers
    if args.num_workers > 0:
        w2v_model = create_shared_word2vec(original_w2v_model)
        use_shared_memory = True
        
        # Set environment variable for workers to find shared memory
        os.environ['SHARED_W2V_NAME'] = w2v_model.shm_name
        print(f"Created shared memory word2vec: {w2v_model.shm_name}")
    else:
        w2v_model = original_w2v_model
        use_shared_memory = False
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
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    # Wrap model for multi-GPU training with fallback strategy
    if args.multi_gpu and not is_distributed:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")
    elif is_distributed:
        # Try DDP first, fallback to DataParallel if it fails
        try:
            # Ensure model is on the correct device before wrapping
            model = model.to(device)
            
            # Create a simple test to verify the model works on this device
            print(f"Testing model on device {device} before DDP wrapping...")
            test_context = torch.randn(1, context_dim).to(device)
            test_word = torch.randn(1, word_onehot_dim).to(device)
            test_gen = torch.randn(1, gen_onehot_dim).to(device)
            
            with torch.no_grad():
                _ = model(test_context, test_word, test_gen)
            torch.cuda.synchronize()
            print(f"Model test passed on rank {local_rank}")
            
            # Clean up test tensors
            del test_context, test_word, test_gen
            torch.cuda.empty_cache()
            
            # Synchronize before wrapping with DDP
            torch.cuda.synchronize()
            dist.barrier()
            
            # Manual parameter broadcast instead of DDP sync
            print(f"Manually broadcasting parameters from rank 0 to rank {local_rank}...")
            
            # Broadcast parameters from rank 0 to all other ranks
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            
            # Broadcast buffers from rank 0 to all other ranks
            for buffer in model.buffers():
                dist.broadcast(buffer.data, src=0)
            
            torch.cuda.synchronize()
            dist.barrier()
            print(f"Manual parameter broadcast completed on rank {local_rank}")
            
            # Now wrap with DDP using process_group=None to skip sync
            print(f"Wrapping with DDP without parameter sync on rank {local_rank}...")
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=False,
                static_graph=False,
                process_group=None  # Skip DDP parameter sync
            )
            print(f"Model wrapped with DistributedDataParallel for GPU {local_rank}")
            
            torch.cuda.synchronize()
            dist.barrier()
            print(f"DDP initialization completed on rank {local_rank}")
            
        except Exception as e:
            print(f"DDP initialization failed on rank {local_rank}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Don't fall back - this should work with these specs
            print(f"Exiting - DDP should work with your hardware specs")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            exit(1)

    if args.train:
        from pathlib import Path
        model_path = args.model
        model_exists = Path(model_path).exists()
        should_resume = False

        if model_exists:
            print(f"Model file '{model_path}' found, checking compatibility...")
            try:
                # Check if it's an ONNX file
                if model_path.endswith('.onnx'):
                    print("ONNX model detected - extracting weights for current model architecture...")
                    try:
                        import onnx
                        import onnxruntime as ort
                        
                        # Load ONNX model to inspect weights
                        onnx_model = onnx.load(model_path)
                        
                        # Get the model's current state dict structure
                        current_state_dict = model.state_dict()
                        
                        # Extract weights from ONNX and map to current model
                        checkpoint = {}
                        
                        # Get ONNX initializers (weights and biases)
                        print(f"ONNX file contains {len(onnx_model.graph.initializer)} initializers")
                        print(f"Current model has {len(current_state_dict)} parameters")
                        
                        for initializer in onnx_model.graph.initializer:
                            param_name = initializer.name
                            param_data = onnx.numpy_helper.to_array(initializer)
                            
                            # Direct name matching first
                            if param_name in current_state_dict:
                                checkpoint[param_name] = torch.from_numpy(param_data.copy())
                            else:
                                # Try common ONNX naming patterns
                                for pytorch_name in current_state_dict.keys():
                                    if param_name.endswith(pytorch_name) or pytorch_name.endswith(param_name):
                                        checkpoint[pytorch_name] = torch.from_numpy(param_data.copy())
                                        break
                                else:
                                    print(f"ONNX parameter '{param_name}' not matched to any model parameter")
                        
                        # Strict parameter matching - must have exact same parameters
                        if len(checkpoint) != len(current_state_dict):
                            print(f"ONNX parameter count mismatch: {len(checkpoint)} vs {len(current_state_dict)} expected")
                            print("Model architecture has changed. Cannot resume training.")
                            exit(1)
                        
                        # Check that all expected parameters are present
                        missing_params = set(current_state_dict.keys()) - set(checkpoint.keys())
                        if missing_params:
                            print(f"Missing parameters in ONNX model: {missing_params}")
                            print("Model architecture has changed. Cannot resume training.")
                            exit(1)
                        
                        # Check for NaN/inf values in loaded weights
                        nan_params = []
                        for name, param in checkpoint.items():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                nan_params.append(name)
                        
                        if nan_params:
                            print(f"CORRUPTED ONNX MODEL: Found NaN/inf values in parameters: {nan_params}")
                            print("The saved model is corrupted. Cannot resume training.")
                            exit(1)
                        
                        print(f"Successfully loaded all {len(checkpoint)} parameters from ONNX model")
                            
                    except ImportError as e:
                        print(f"Missing required library for ONNX loading: {e}")
                        print("Please install: pip install onnx onnxruntime")
                        print("Starting training from scratch instead...")
                        should_resume = False
                    except Exception as e:
                        print(f"Error loading ONNX model: {e}")
                        print("Starting training from scratch instead...")
                        should_resume = False
                else:
                    # Load PyTorch model
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

        # Create dataset with parallel processing
        print(f"Creating dataset with {args.num_workers} workers for initialization...")
        dataset = CharGenStreamingDataset(
            args.data_dir, 
            w2v_model, 
            char_to_id,
            ctx_len=args.ctx_len, 
            max_word_len=args.max_word_len, 
            max_gen_len=args.max_gen_len,
            num_workers=args.num_workers,
            use_shared_memory=use_shared_memory
        )
        
        # Setup data loading based on distributed mode
        if is_distributed:
            # IterableDataset doesn't use samplers - data distribution is handled by workers reading different files
            workers_per_process = max(1, args.num_workers // max(1, torch.cuda.device_count()))
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                num_workers=workers_per_process,
                pin_memory=True,
                persistent_workers=True if workers_per_process > 0 else False,
                prefetch_factor=2,
                drop_last=True  # Ensure consistent batch sizes across ranks
            )
            print(f"Rank {local_rank}: Distributed streaming DataLoader created with {workers_per_process} workers per process")
        else:
            # Regular DataLoader for single GPU or DataParallel
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            print(f"Streaming DataLoader created with {args.num_workers} worker processes")
        
        # Train the model (update is_distributed flag based on actual state)
        train_model(
            model, 
            dataloader, 
            vocab_size=char_vocab_size, 
            epochs=args.epochs, 
            save_path=args.model,
            local_rank=local_rank if is_distributed else 0,
            is_distributed=is_distributed,
            num_workers=args.num_workers,
            total_samples=getattr(dataset, 'total_samples', None),
            batch_size=args.batch_size,
            context_dim=context_dim,
            word_onehot_dim=word_onehot_dim,
            gen_onehot_dim=gen_onehot_dim
        )
        
        # Cleanup shared memory after training
        if use_shared_memory:
            cleanup_shared_word2vec()
        
    elif args.predict:
        # For prediction, we need to unwrap the model if it's wrapped in DataParallel or DDP
        if isinstance(model, (nn.DataParallel, DDP)):
            model = model.module
        
        # Load the model weights
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
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
        
        # Cleanup shared memory on exit
        if use_shared_memory:
            cleanup_shared_word2vec()
