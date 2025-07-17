import os
import re
import json
import random
import torch
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import psutil
import sys

# -------- CONFIG --------
INPUT_FILE = "./wiki_sentences.txt"  # Output from extract_sentences_wiki.py
OUTPUT_FILE = "./wiki_training_data.optimized.pt"
MIN_LEN = 2     # minimum number of tokens to include a prefix
MAX_LEN = 100   # max number of tokens per prefix
ERRORS_PER_SAMPLE = 2  # Number of error samples per prefix
SELECTION_PROB = 1.0   # Probability of selecting a sentence (1.0 = use all sentences)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
CHUNK_SIZE = 10000  # Number of lines to process in each batch
NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
# ------------------------

def smart_tokenize(text):
    """
    Tokenize a sentence into words and punctuation.
    Keeps contractions like "don't" together.
    """
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

def random_corrupt(word):
    """Introduce a random error in the word, with 20% chance of no error"""
    # 20% chance to return the word unchanged (no error)
    if random.random() < 0.2:
        return word
        
    if len(word) == 0:
        return word
    op = random.choice(["insert", "delete", "substitute", "transpose"])
    i = random.randint(0, len(word) - 1)

    if op == "insert":
        c = random.choice(ALPHABET)
        return word[:i] + c + word[i:]

    elif op == "delete" and len(word) > 1:
        return word[:i] + word[i+1:]

    elif op == "substitute":
        c = random.choice(ALPHABET.replace(word[i], ''))
        return word[:i] + c + word[i+1:]

    elif op == "transpose" and len(word) > 1 and i < len(word) - 1:
        return word[:i] + word[i+1] + word[i] + word[i+2:]

    else:
        return word

def generate_error_samples_from_sentence(sentence):
    """Generate samples with random context lengths for a sentence"""
    tokens = smart_tokenize(sentence)
    
    # Skip sentences that are too short or too long
    if len(tokens) < MIN_LEN or len(tokens) > MAX_LEN:
        return []
    
    samples = []
    max_target_pos = min(len(tokens), 11)  # position after context
    
    for i in range(MIN_LEN-1, max_target_pos):
        target_word = tokens[i]
        
        # Skip very short target words
        if len(target_word) < 2:
            continue
        
        # Choose 2 random context lengths for this target
        possible_lengths = list(range(min(i+1, 11)))  # +1 because we want to include full length
        if len(possible_lengths) > 2:
            context_lengths = random.sample(possible_lengths, 2)
        else:
            context_lengths = possible_lengths  # use all if less than 2 available
        
        # Generate samples for each chosen context length
        for ctx_len in context_lengths:
            prefix_tokens = tokens[max(0, i-ctx_len):i]  # take last ctx_len tokens as context
            
            # Generate error samples for this context
            used = set()
            error_samples = []
            
            # Try to generate exactly ERRORS_PER_SAMPLE versions
            attempts = 0
            max_attempts = 20  # Limit attempts to avoid infinite loops
            
            while len(error_samples) < ERRORS_PER_SAMPLE and attempts < max_attempts:
                corrupted = random_corrupt(target_word)
                attempts += 1
                
                # Avoid duplicates
                if corrupted not in used:
                    used.add(corrupted)
                    error_samples.append({
                        "context": prefix_tokens,
                        "misspelled": corrupted,
                        "target": target_word
                    })
            
            # If we couldn't generate enough unique versions, fill with additional samples
            while len(error_samples) < ERRORS_PER_SAMPLE:
                corrupted = random_corrupt(target_word)
                error_samples.append({
                    "context": prefix_tokens,
                    "misspelled": corrupted,
                    "target": target_word
                })
            
            samples.extend(error_samples)
    
    return samples

def process_chunk(chunk):
    """Process a chunk of sentences and return all error samples"""
    all_samples = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        samples = generate_error_samples_from_sentence(line)
        all_samples.extend(samples)
    return all_samples

def count_lines(file_path):
    """Count lines in a file efficiently"""
    print(f"Counting lines in {file_path}...")
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # Count total lines for progress tracking
    total_lines = count_lines(INPUT_FILE)
    print(f"📊 Found {total_lines:,} sentences in input file")
    
    if SELECTION_PROB < 1.0:
        estimated_lines = int(total_lines * SELECTION_PROB)
        print(f"🔄 Using selection probability of {SELECTION_PROB:.2f}, expecting ~{estimated_lines:,} sentences")
    
    # Check available memory
    try:
        mem_info = psutil.virtual_memory()
        print(f"System memory: {mem_info.total / (1024**3):.1f} GB total, "
              f"{mem_info.available / (1024**3):.1f} GB available")
    except:
        print("Could not get system memory information")
    
    # Pre-allocate lists with estimated size
    total_lines = count_lines(INPUT_FILE)
    estimated_selected = int(total_lines * SELECTION_PROB)
    # Each selected sentence can generate up to 10 contexts, each with ERRORS_PER_SAMPLE variations
    estimated_samples = estimated_selected * 10 * ERRORS_PER_SAMPLE
    
    print(f"Pre-allocating lists for estimated {estimated_samples:,} samples...")
    contexts = [None] * estimated_samples  # Pre-allocate with None
    misspelled = [None] * estimated_samples
    targets = [None] * estimated_samples
    
    # Process file in chunks to manage memory
    with open(INPUT_FILE, "r", encoding="utf-8") as in_file:
        chunk = []
        total_samples = 0
        processed_lines = 0
        selected_lines = 0
        
        print(f"🚀 Generating samples using {NUM_WORKERS} worker processes")
        pbar = tqdm(total=total_lines, desc="Processing sentences")
        
        for line in in_file:
            processed_lines += 1
            
            # Apply selection probability
            if random.random() <= SELECTION_PROB:
                chunk.append(line)
                selected_lines += 1
            
            # Process when chunk is full or at end of file
            if len(chunk) >= CHUNK_SIZE:
                # Split chunk into sub-chunks for parallel processing
                sub_chunks = [chunk[i:i+CHUNK_SIZE//NUM_WORKERS] for i in range(0, len(chunk), CHUNK_SIZE//NUM_WORKERS)]
                
                # Process sub-chunks in parallel
                with mp.Pool(NUM_WORKERS) as pool:
                    results = pool.map(process_chunk, sub_chunks)
                
                # Collect results
                for sample_list in results:
                    for sample in sample_list:
                        if total_samples < estimated_samples:
                            contexts[total_samples] = sample["context"]
                            misspelled[total_samples] = sample["misspelled"]
                            targets[total_samples] = sample["target"]
                            total_samples += 1
                        else:
                            # If we exceeded estimate, extend lists
                            contexts.append(sample["context"])
                            misspelled.append(sample["misspelled"])
                            targets.append(sample["target"])
                            total_samples += 1
                
                # Update progress
                pbar.update(len(chunk))
                chunk = []
                
                # Show current stats
                pbar.set_postfix(samples=f"{total_samples:,}")
                
                # Report memory usage periodically
                if processed_lines % 50000 == 0:
                    try:
                        mem_info = psutil.virtual_memory()
                        print(f"Memory: {mem_info.percent}% used, {mem_info.available / (1024**3):.1f} GB available")
                    except:
                        pass
        
        # Process remaining lines
        if chunk:
            sub_chunks = [chunk[i:i+max(1, len(chunk)//NUM_WORKERS)] for i in range(0, len(chunk), max(1, len(chunk)//NUM_WORKERS))]
            
            with mp.Pool(NUM_WORKERS) as pool:
                results = pool.map(process_chunk, sub_chunks)
            
            for sample_list in results:
                for sample in sample_list:
                    contexts.append(sample["context"])
                    misspelled.append(sample["misspelled"])
                    targets.append(sample["target"])
                    total_samples += 1
            
            pbar.update(len(chunk))
        
        pbar.close()
    
    # Create sample offsets for efficient indexing
    print("Building index offsets...")
    sample_offsets = [0]
    total_expanded_samples = 0
    
    for target in tqdm(targets, desc="Building offsets"):
        # Each target word generates len(target) + 1 training examples
        # (one for each prefix length, plus one for the <eow> token)
        prefix_count = len(target) + 1
        total_expanded_samples += prefix_count
        sample_offsets.append(total_expanded_samples)
    
    # Save the optimized data
    print(f"💾 Saving optimized data to {OUTPUT_FILE}...")
    torch.save({
        'contexts': contexts,
        'misspelled': misspelled,
        'targets': targets,
        'sample_offsets': sample_offsets,
        'total_samples': total_expanded_samples
    }, OUTPUT_FILE)
    
    print(f"\n✅ Selected {selected_lines:,} out of {processed_lines:,} sentences ({selected_lines/processed_lines*100:.1f}%)")
    print(f"📊 Generated {total_samples:,} samples")
    print(f"🔢 Will produce {total_expanded_samples:,} training examples")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    
    # Print file size info
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"📏 Output file size: {file_size / (1024*1024):.1f} MB")

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
    
    parser = argparse.ArgumentParser(description="Generate optimized training samples directly from sentences")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input file with sentences")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file for optimized data")
    parser.add_argument("--min-len", type=int, default=MIN_LEN, help="Minimum tokens in prefix")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="Maximum tokens in prefix")
    parser.add_argument("--errors", type=int, default=ERRORS_PER_SAMPLE, help="Number of error samples per prefix")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for processing")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    
    parser.add_argument("--selection-prob", type=float, default=SELECTION_PROB, 
                        help="Probability of selecting each sentence (1.0 = use all sentences)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    MIN_LEN = args.min_len
    MAX_LEN = args.max_len
    ERRORS_PER_SAMPLE = args.errors
    CHUNK_SIZE = args.chunk_size
    NUM_WORKERS = args.workers
    SELECTION_PROB = args.selection_prob
    
    main()
