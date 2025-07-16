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
import signal
import atexit
from batch_sampler import BatchSamplerByChunks
from functools import lru_cache
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import math

# Global list to track all created processes for cleanup
_all_processes = []

def cleanup_processes():
    """Clean up any remaining processes on exit"""
    for p in _all_processes:
        if p.is_alive():
            print(f"Terminating process {p.pid} during cleanup")
            try:
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    print(f"Process {p.pid} still alive after terminate, killing...")
                    os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error terminating process {p.pid}: {e}")

# Register cleanup function
atexit.register(cleanup_processes)

# Handle termination signals
def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up processes...")
    cleanup_processes()
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---- Parallel Index Building Functions ----

def count_file_lines(filename):
    """Count the number of lines in a file efficiently"""
    with open(filename, 'rb') as f:
        # Count newline characters
        return sum(1 for _ in f)

def process_chunk(chunk_info):
    """Process a chunk of the file to build partial index"""
    filename, start_line, end_line, chunk_id = chunk_info
    
    # Print worker information for debugging
    worker_id = os.getpid()
    print(f"Worker {worker_id} (chunk {chunk_id}) processing lines {start_line}-{end_line}")
    
    offsets = []
    sample_lengths = []
    total_expanded_samples = 0
    
    try:
        # Open file with a large buffer to reduce I/O operations
        with open(filename, 'rb', buffering=8*1024*1024) as f:
            # Skip to start_line - use efficient skipping
            if start_line > 0:
                # First try to seek to an approximate position
                # Estimate bytes per line from first few lines
                sample_size = 1000
                f.seek(0)
                sample_data = f.read(sample_size)
                newline_count = sample_data.count(b'\n')
                if newline_count > 0:
                    bytes_per_line_estimate = sample_size / newline_count
                    # Seek to estimated position
                    f.seek(int(start_line * bytes_per_line_estimate))
                    # Count lines to adjust position
                    line_count = 0
                    while True:
                        if f.readline():
                            line_count += 1
                        else:
                            break
                    # Seek back to beginning and skip lines properly
                    f.seek(0)
                    lines_to_skip = max(0, start_line - line_count)
                else:
                    lines_to_skip = start_line
                
                # Skip remaining lines
                for _ in range(lines_to_skip):
                    f.readline()
            
            # Process assigned chunk
            pos = f.tell()
            lines_processed = 0
            
            # Read in larger blocks for efficiency
            while lines_processed < (end_line - start_line):
                # Read a block of lines
                block_size = min(1000, (end_line - start_line) - lines_processed)
                block_lines = []
                block_positions = []
                
                for _ in range(block_size):
                    block_positions.append(pos)
                    line = f.readline()
                    if not line:  # End of file
                        break
                    block_lines.append(line)
                    pos += len(line)
                
                if not block_lines:
                    break
                
                # Process the block
                for i, line in enumerate(block_lines):
                    offsets.append(block_positions[i])
                    
                    # Count prefix samples this line will generate
                    try:
                        sample = json.loads(line.strip().decode('utf-8'))
                        target = sample["target"]
                        prefix_count = len(target) + 1  # +1 for the <eow> case
                        sample_lengths.append(prefix_count)
                        total_expanded_samples += prefix_count
                    except Exception as e:
                        # Handle corrupted lines
                        sample_lengths.append(1)
                        total_expanded_samples += 1
                
                lines_processed += len(block_lines)
                
                # Periodically report progress
                if lines_processed % 10000 == 0:
                    print(f"Worker {worker_id} (chunk {chunk_id}): processed {lines_processed}/{end_line-start_line} lines")
        
        # Print completion information
        print(f"Worker {worker_id} completed chunk {chunk_id}: {len(offsets)} samples, {total_expanded_samples} expanded samples")
        
        return chunk_id, offsets, sample_lengths, total_expanded_samples
    
    except Exception as e:
        print(f"ERROR in worker {worker_id} processing chunk {chunk_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return chunk_id, [], [], 0
def process_chunk_wrapper(chunk, result_queue):
    """Wrapper function to put results in queue"""
    # Set lower process priority
    try:
        import resource
        # Check if the function exists before calling it
        if hasattr(resource, 'nice') and callable(resource.nice):
            resource.nice(10)
        elif hasattr(os, 'nice') and callable(os.nice):
            os.nice(10)
    except (ImportError, AttributeError, PermissionError):
        # Silently continue if the function is not available
        pass
    
    # Process the chunk
    result = process_chunk(chunk)
    result_queue.put(result)

# Function to run in a separate process
def parallel_index_builder(json_path, num_workers):
    """Build index in parallel using direct process creation"""
    print(f"Starting parallel index builder with {num_workers} workers")
    
    # Count total lines
    total_lines = count_file_lines(json_path)
    print(f"Found {total_lines:,} lines in dataset")
    
    # Split the file into chunks
    chunk_size = math.ceil(total_lines / num_workers)
    chunks = [(json_path, i * chunk_size, min((i + 1) * chunk_size, total_lines), i) 
             for i in range(num_workers)]
    
    # Create processes
    processes = []
    result_queue = mp.Queue()
    
    # Set CPU affinity if possible (Linux only)
    try:
        import psutil
        p = psutil.Process()
        # Get available CPUs
        available_cpus = list(range(psutil.cpu_count()))
        print(f"Available CPUs: {available_cpus}")
    except ImportError:
        print("psutil not available, skipping CPU affinity setting")
        available_cpus = None
    
    # Start processes with specific CPU affinity if possible
    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=process_chunk_wrapper,
            args=(chunk, result_queue),
            daemon=True  # Set as daemon so it terminates when main process exits
        )
        
        # Try to set CPU affinity if psutil is available
        if available_cpus and i < len(available_cpus):
            try:
                # Check if the function exists before calling it
                if hasattr(p, 'cpu_affinity') and callable(p.cpu_affinity):
                    p.cpu_affinity([available_cpus[i % len(available_cpus)]])
                    print(f"Set process {i} to CPU {available_cpus[i % len(available_cpus)]}")
            except (AttributeError, ImportError, PermissionError):
                # cpu_affinity might not be available on all platforms
                print("CPU affinity setting not supported")
        
        # Add to global process list for cleanup
        _all_processes.append(p)
        
        processes.append(p)
        p.start()
        print(f"Started process {p.pid} for chunk {i}")
    
    # Collect results with progress bar
    results = []
    for _ in tqdm(range(len(chunks)), desc=f"Collecting results from {num_workers} workers"):
        results.append(result_queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            print(f"Warning: Process {p.pid} did not terminate, forcing termination")
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                print(f"Error: Could not terminate process {p.pid}")
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except:
                    pass
    
    # Remove processes from global list
    for p in processes:
        if p in _all_processes:
            _all_processes.remove(p)
    
    print(f"All {len(processes)} processes completed")
    
    # Sort results by chunk_id
    results.sort(key=lambda x: x[0])
    
    # Process results
    all_offsets = []
    all_sample_lengths = []
    total_expanded_samples = 0
    
    for chunk_id, offsets, sample_lengths, expanded_samples in results:
        # Adjust offsets for chunks after the first one
        if chunk_id > 0 and offsets:
            # Calculate the correct file position
            chunk_start_line = chunks[chunk_id][1]  # Start line for this chunk
            with open(json_path, 'rb') as f:
                # Skip to the start line of this chunk
                for _ in range(chunk_start_line):
                    f.readline()
                # This is the actual position in the file
                actual_pos = f.tell()
                # Adjust all offsets in this chunk
                offsets = [pos - offsets[0] + actual_pos for pos in offsets]
        
        all_offsets.extend(offsets)
        all_sample_lengths.extend(sample_lengths)
        total_expanded_samples += expanded_samples
    
    # Build cumulative index
    cumulative_lengths = []
    cumsum = 0
    for length in tqdm(all_sample_lengths, desc="Building cumulative index"):
        cumsum += length
        cumulative_lengths.append(cumsum)
    
    return all_offsets, all_sample_lengths, cumulative_lengths, total_expanded_samples

# ---- Char Map ----

def create_charmap():
    printable = string.printable.replace('"', '')  # Avoid JSON quote issues
    char_list = list(printable) + ["<eow>"]
    char_to_id = {c: i for i, c in enumerate(char_list)}
    id_to_char = {i: c for i, c in enumerate(char_list)}
    return char_to_id, id_to_char

# ---- Input Preparation ----

def pad_context(context_words, ctx_len=10):
    """Pad or truncate context words to the specified length"""
    if len(context_words) > ctx_len:
        return context_words[-ctx_len:]  # Take last ctx_len words
    return context_words  # Will be padded with zeros in vectorize_context

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
    def __init__(self, json_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50, num_workers=None):
        self.json_path = json_path
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len
        
        # Use number of CPU cores if not specified
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 1)
        
        print(f"🔄 Using {self.num_workers} workers for index operations")
        print(f"🔄 System has {mp.cpu_count()} CPU cores available")
        
        # Check if cached index exists
        index_path = f"{json_path}.index"
        
        if os.path.exists(index_path):
            print(f"📂 Loading cached index from {index_path}")
            try:
                # Use direct multiprocessing for index loading
                self.load_index_direct_mp(index_path)
                print(f"✅ Loaded index: {self.total_samples:,} samples from {len(self.offsets):,} original samples")
                return
            except Exception as e:
                print(f"⚠️ Failed to load cached index: {e}")
                import traceback
                traceback.print_exc()
                print("Building new index...")

        # Use completely separate process for index building
        print(f"Starting index building with {self.num_workers} workers...")
        
        # Build index using our parallel implementation
        all_offsets, all_sample_lengths, cumulative_lengths, total_expanded_samples = parallel_index_builder(
            json_path, self.num_workers
        )
        
        self.offsets = all_offsets
        self.sample_lengths = all_sample_lengths
        self.cumulative_lengths = cumulative_lengths
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
    def load_index_direct_mp(self, index_path):
        """Load index using direct multiprocessing with explicit process creation"""
        print(f"Loading index with direct multiprocessing using {self.num_workers} workers...")
        
        # Initialize empty lists for results
        self.offsets = []
        self.sample_lengths = []
        self.cumulative_lengths = []
        
        # First load metadata to get sizes
        print("Loading metadata...")
        with open(index_path, 'rb') as f:
            # Load just enough to get the metadata
            metadata = torch.load(f)
            total_samples = metadata['total_samples']
            offsets_size = len(metadata['offsets'])
            
            print(f"Index contains {offsets_size:,} samples, {total_samples:,} expanded samples")
            
            # Clear metadata to free memory
            del metadata
        
        # Calculate chunk sizes - use more chunks than workers to reduce memory per process
        num_chunks = self.num_workers * 3  # Use more chunks to reduce memory per process
        chunk_size = max(1, offsets_size // num_chunks)
        
        print(f"Starting with {num_chunks} chunks, chunk size: {chunk_size:,}")
        
        # Create overall progress bar for all chunks
        overall_progress = tqdm(
            total=offsets_size,
            desc="Overall index loading progress",
            unit="samples"
        )
        
        try:
            # Process chunks in batches to limit memory usage
            batch_ranges = list(range(0, num_chunks, self.num_workers))
            for batch_idx, batch_start in enumerate(batch_ranges):
                batch_end = min(batch_start + self.num_workers, num_chunks)
                active_workers = batch_end - batch_start
                
                print(f"Processing batch {batch_idx+1}/{len(batch_ranges)}: {active_workers} chunks ({batch_start} to {batch_end-1})")
                
                # Clear previous batch data
                processes = []
                pipes = []
                
                # Create and start processes for this batch
                for i in range(batch_start, batch_end):
                    # Create pipe for this process
                    parent_conn, child_conn = mp.Pipe()
                    pipes.append(parent_conn)
                    
                    # Calculate chunk range
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, offsets_size)
                    
                    if start_idx >= end_idx:
                        continue
                        
                    # Create and start process with daemon=True
                    p = mp.Process(
                        target=self._load_index_chunk_process,
                        args=(index_path, start_idx, end_idx, i, child_conn),
                        daemon=True  # Set as daemon so it terminates when main process exits
                    )
                    
                    # Add to global process list for cleanup
                    _all_processes.append(p)
                    
                    processes.append(p)
                    p.start()
                    print(f"Started process {p.pid} for chunk {i} ({start_idx:,} to {end_idx:,})")
                
                # Create progress bar for this batch
                batch_progress = tqdm(
                    total=len(processes),
                    desc=f"Batch {batch_idx+1}/{len(batch_ranges)}",
                    unit="chunks"
                )
                
                # Collect results from all processes in this batch with timeout
                batch_results = []
                for i, pipe in enumerate(pipes):
                    if i >= len(processes):
                        continue
                    
                    try:
                        # Set a timeout for receiving data
                        if pipe.poll(30):  # Wait up to 30 seconds
                            result = pipe.recv()
                            batch_results.append(result)
                            # Update progress bars
                            batch_progress.update(1)
                            chunk_id, offsets, sample_lengths, _ = result
                            overall_progress.update(len(offsets))
                        else:
                            print(f"Timeout waiting for process {batch_start + i}")
                    except EOFError:
                        print(f"Error: Process {batch_start + i} closed pipe unexpectedly")
                
                # Close batch progress bar
                batch_progress.close()
                
                # Wait for all processes in this batch to finish
                for p in processes:
                    p.join(timeout=5)  # Wait up to 5 seconds
                    if p.is_alive():
                        print(f"Warning: Process {p.pid} did not terminate, forcing termination")
                        p.terminate()
                        p.join(timeout=1)
                        if p.is_alive():
                            print(f"Error: Could not terminate process {p.pid}")
                            try:
                                os.kill(p.pid, signal.SIGKILL)
                            except:
                                pass
                
                # Remove processed workers from global list
                for p in processes:
                    if p in _all_processes:
                        _all_processes.remove(p)
                
                # Process batch results
                for chunk_id, offsets, sample_lengths, cumulative_lengths in batch_results:
                    if not offsets:  # Skip empty results
                        continue
                        
                    self.offsets.extend(offsets)
                    self.sample_lengths.extend(sample_lengths)
                    
                    # Adjust cumulative lengths for proper concatenation
                    if self.cumulative_lengths and cumulative_lengths:
                        base = self.cumulative_lengths[-1]
                        self.cumulative_lengths.extend([cl + base for cl in cumulative_lengths])
                    else:
                        self.cumulative_lengths.extend(cumulative_lengths)
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                print(f"Completed batch {batch_idx+1}/{len(batch_ranges)}, processed {len(self.offsets):,} samples so far")
            
            # Close overall progress bar
            overall_progress.close()
            
            # Set total samples
            self.total_samples = total_samples
            print(f"All chunks processed successfully")
            
        except Exception as e:
            # Close progress bars in case of error
            overall_progress.close()
            
            print(f"Error during index loading: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up any remaining processes
            for p in processes:
                if p.is_alive():
                    print(f"Terminating process {p.pid}")
                    p.terminate()
                    # Remove from global list
                    if p in _all_processes:
                        _all_processes.remove(p)
            
            raise
    
    @staticmethod
    def _load_index_chunk_process(index_path, start_idx, end_idx, chunk_id, conn):
        """Process function to load a chunk of the index in a separate process"""
        try:
            # Set lower memory priority for this process - handle missing functions
            try:
                import resource
                # Check if the function exists before calling it
                if hasattr(resource, 'setpriority') and hasattr(resource, 'PRIO_PROCESS'):
                    resource.setpriority(resource.PRIO_PROCESS, os.getpid(), 10)
            except (ImportError, AttributeError, PermissionError):
                # Silently continue if the function is not available
                pass
                
            # Print process info
            pid = os.getpid()
            print(f"Process {pid} (chunk {chunk_id}) starting: loading {start_idx:,} to {end_idx:,}")
            
            # Load only the needed part of the index to reduce memory usage
            offsets = []
            sample_lengths = []
            cumulative_lengths = []
            
            # Create a progress bar for this worker
            worker_progress = tqdm(
                total=100,  # We'll update based on percentage
                desc=f"Worker {pid} (chunk {chunk_id})",
                position=chunk_id % 10,  # Stagger progress bars
                leave=False  # Don't leave the progress bar when done
            )
            
            try:
                # First try the optimized approach
                import pickle
                with open(index_path, 'rb') as f:
                    try:
                        # Update progress
                        worker_progress.update(10)
                        
                        # Skip the PyTorch header
                        magic_number = pickle.load(f)
                        protocol_version = pickle.load(f)
                        sys_info = pickle.load(f)
                        
                        # Update progress
                        worker_progress.update(20)
                        
                        # Load the actual data dictionary
                        data = pickle.load(f)
                        
                        # Update progress
                        worker_progress.update(30)
                        
                        # Extract only the needed portions
                        if 'offsets' in data:
                            offsets = data['offsets'][start_idx:end_idx]
                            worker_progress.update(10)
                        
                        if 'sample_lengths' in data:
                            sample_lengths = data['sample_lengths'][start_idx:end_idx]
                            worker_progress.update(10)
                        
                        # For cumulative lengths, we need to adjust based on the chunk
                        if 'cumulative_lengths' in data:
                            if start_idx == 0:
                                # First chunk, take as is
                                cumulative_lengths = data['cumulative_lengths'][start_idx:end_idx]
                            elif start_idx < len(data['cumulative_lengths']):
                                # Subsequent chunks, adjust to start from 0
                                prev_cumulative = data['cumulative_lengths'][start_idx - 1]
                                cumulative_lengths = [
                                    cl - prev_cumulative for cl in data['cumulative_lengths'][start_idx:end_idx]
                                ]
                            worker_progress.update(10)
                        
                        # Clear data to free memory
                        del data
                        worker_progress.update(10)
                    except Exception as e:
                        print(f"Error with optimized loading: {e}, falling back to torch.load")
                        raise
            except Exception:
                # Fall back to standard torch.load if the optimized approach fails
                print(f"Process {pid} falling back to standard torch.load")
                try:
                    # Reset progress
                    worker_progress.reset()
                    
                    # Use torch.load with map_location='cpu' to ensure it loads on CPU
                    index_data = torch.load(index_path, map_location='cpu')
                    worker_progress.update(50)
                    
                    # Extract the relevant chunks
                    offsets = index_data['offsets'][start_idx:end_idx]
                    worker_progress.update(10)
                    
                    sample_lengths = index_data['sample_lengths'][start_idx:end_idx]
                    worker_progress.update(10)
                    
                    # For cumulative lengths, we need to adjust based on the chunk
                    if start_idx == 0:
                        # First chunk, take as is
                        cumulative_lengths = index_data['cumulative_lengths'][start_idx:end_idx]
                    elif start_idx < len(index_data['cumulative_lengths']):
                        # Subsequent chunks, adjust to start from 0
                        prev_cumulative = index_data['cumulative_lengths'][start_idx - 1]
                        cumulative_lengths = [
                            cl - prev_cumulative for cl in index_data['cumulative_lengths'][start_idx:end_idx]
                        ]
                    worker_progress.update(20)
                    
                    # Clear data to free memory
                    del index_data
                    worker_progress.update(10)
                except Exception as e:
                    print(f"Both loading methods failed: {e}")
                    raise
            
            # Force garbage collection before sending results
            import gc
            gc.collect()
            
            # Complete the progress bar
            worker_progress.close()
            
            # Send results back through the pipe
            conn.send((chunk_id, offsets, sample_lengths, cumulative_lengths))
            
            print(f"Process {pid} (chunk {chunk_id}) completed: processed {len(offsets):,} samples")
            
        except Exception as e:
            print(f"ERROR in process {os.getpid()} (chunk {chunk_id}): {str(e)}")
            import traceback
            traceback.print_exc()
            # Send empty results in case of error
            try:
                conn.send((chunk_id, [], [], []))
            except:
                pass
        finally:
            # Close the connection
            try:
                conn.close()
            except:
                pass
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
    
    # Learning rate scheduler - removed verbose parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
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
    # Force multiprocessing to use spawn method for better compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
    
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
    parser.add_argument("--index_workers", type=int, default=8, help="Number of worker processes for index building")
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

        # Print the number of workers being used
        print(f"🔧 Using {args.index_workers} workers for index building")
        print(f"🔧 Using {args.num_workers} workers for data loading")
        
        # Now proceed to dataset and training as before
        dataset = CharGenLazyDataset(args.data, w2v_model, char_to_id,
                                ctx_len=args.ctx_len, max_word_len=args.max_word_len, 
                                max_gen_len=args.max_gen_len, num_workers=args.index_workers)
        
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
