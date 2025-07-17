import os
import re
import json
import random
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import psutil
import sys

# -------- CONFIG --------
INPUT_FILE = "./wiki_prefix_samples.txt"
OUTPUT_FILE = "./wiki_error_samples.optimized.pt"
ERRORS_PER_SAMPLE = 2  # Reduced from 10 to 2 for Wikipedia data
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
NUM_WORKERS = cpu_count()  # or set manually
CHUNK_SIZE = 1000  # Process in chunks for better progress reporting
# ------------------------

def smart_tokenize(text):
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

def generate_error_samples(line):
    """Generate exactly ERRORS_PER_SAMPLE error samples per prefix"""
    tokens = smart_tokenize(line)
    if len(tokens) < 2:
        return []

    last_word = tokens[-1]
    prefix = tokens[:-1]

    samples = []
    used = set()

    # Try to generate exactly ERRORS_PER_SAMPLE versions (which may include unchanged words)
    attempts = 0
    max_attempts = 20  # Limit attempts to avoid infinite loops
    
    while len(samples) < ERRORS_PER_SAMPLE and attempts < max_attempts:
        corrupted = random_corrupt(last_word)
        attempts += 1
        
        # Now we accept both changed and unchanged words, but avoid duplicates
        if corrupted not in used:
            used.add(corrupted)
            samples.append({
                "context": prefix,
                "misspelled": corrupted,
                "target": last_word
            })

    # If we couldn't generate enough unique versions, fill with additional samples
    while len(samples) < ERRORS_PER_SAMPLE:
        corrupted = random_corrupt(last_word)
        samples.append({
            "context": prefix,
            "misspelled": corrupted,
            "target": last_word
        })

    return samples

def process_chunk(chunk):
    """Process a chunk of lines and return all samples"""
    all_samples = []
    for line in chunk:
        if line.strip():
            samples = generate_error_samples(line.strip())
            all_samples.extend(samples)
    return all_samples

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print(f"🔍 Reading input file: {INPUT_FILE}")
    
    # Count lines for progress tracking
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"📊 Found {total_lines:,} prefixes to process")
    print(f"⚙️ Will generate {ERRORS_PER_SAMPLE} error samples per prefix")
    print(f"🧮 Expected output: ~{total_lines * ERRORS_PER_SAMPLE:,} samples")
    
    # Check available memory
    try:
        mem_info = psutil.virtual_memory()
        print(f"System memory: {mem_info.total / (1024**3):.1f} GB total, "
              f"{mem_info.available / (1024**3):.1f} GB available")
    except:
        print("Could not get system memory information")
    
    # Store data in memory-efficient structures
    contexts = []  # List of context word lists
    misspelled = []  # List of misspelled words
    targets = []  # List of target words
    
    # Process in chunks for better memory management
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines_processed = 0
        samples_generated = 0
        
        with Pool(NUM_WORKERS) as pool:
            pbar = tqdm(total=total_lines, desc="Processing prefixes")
            
            # Process file in chunks
            while True:
                chunk = [f.readline().strip() for _ in range(CHUNK_SIZE)]
                chunk = [line for line in chunk if line]  # Remove empty lines
                
                if not chunk:
                    break
                
                # Process chunk in parallel
                results = pool.map(generate_error_samples, chunk)
                
                # Collect results
                for sample_list in results:
                    for sample in sample_list:
                        contexts.append(sample["context"])
                        misspelled.append(sample["misspelled"])
                        targets.append(sample["target"])
                        samples_generated += 1
                
                lines_processed += len(chunk)
                pbar.update(len(chunk))
                pbar.set_postfix(samples=f"{samples_generated:,}")
                
                # Report memory usage periodically
                if lines_processed % 10000 == 0:
                    try:
                        mem_info = psutil.virtual_memory()
                        print(f"Memory: {mem_info.percent}% used, {mem_info.available / (1024**3):.1f} GB available")
                    except:
                        pass
            
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
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Processed {lines_processed:,} prefixes")
    print(f"📝 Generated {samples_generated:,} samples")
    print(f"🔢 Will produce {total_expanded_samples:,} training examples")
    print(f"📁 Output saved to: {OUTPUT_FILE}")
    
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
    
    parser = argparse.ArgumentParser(description="Generate optimized training samples with artificial errors")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input file with prefixes")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output optimized PT file")
    parser.add_argument("--errors", type=int, default=ERRORS_PER_SAMPLE, help="Number of error samples per prefix")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    ERRORS_PER_SAMPLE = args.errors
    NUM_WORKERS = args.workers
    CHUNK_SIZE = args.chunk_size
    
    main()
