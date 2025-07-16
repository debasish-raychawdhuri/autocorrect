import os
import re
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# -------- CONFIG --------
INPUT_FILE = "./wiki_sentences.txt"  # Output from extract_sentences_wiki.py
OUTPUT_FILE = "./wiki_prefix_samples.txt"
MIN_LEN = 2     # minimum number of tokens to include a prefix
MAX_LEN = 100   # max number of tokens per prefix
CHUNK_SIZE = 100000  # Number of lines to process in each batch
NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
# ------------------------

def smart_tokenize(text):
    """
    Tokenize a sentence into words and punctuation.
    Keeps contractions like "don't" together.
    """
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

def generate_left_contexts(sentence):
    """Generate all valid prefixes for a sentence"""
    tokens = smart_tokenize(sentence)
    prefixes = [" ".join(tokens[:i+1]) for i in range(len(tokens))]
    # Filter out very short or long prefixes
    return [p for p in prefixes if len(p.split()) >= MIN_LEN and len(p.split()) <= MAX_LEN]

def process_chunk(chunk):
    """Process a chunk of sentences and return all prefixes"""
    all_prefixes = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        prefixes = generate_left_contexts(line)
        all_prefixes.extend(prefixes)
    return all_prefixes

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
    
    # Process file in chunks to manage memory
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        with open(INPUT_FILE, "r", encoding="utf-8") as in_file:
            # Process in chunks
            chunk = []
            total_prefixes = 0
            processed_lines = 0
            
            print(f"🚀 Generating prefixes using {NUM_WORKERS} worker processes")
            pbar = tqdm(total=total_lines, desc="Processing sentences")
            
            for line in in_file:
                chunk.append(line)
                processed_lines += 1
                
                # Process when chunk is full or at end of file
                if len(chunk) >= CHUNK_SIZE:
                    # Split chunk into sub-chunks for parallel processing
                    sub_chunks = [chunk[i:i+CHUNK_SIZE//NUM_WORKERS] for i in range(0, len(chunk), CHUNK_SIZE//NUM_WORKERS)]
                    
                    # Process sub-chunks in parallel
                    with mp.Pool(NUM_WORKERS) as pool:
                        results = pool.map(process_chunk, sub_chunks)
                    
                    # Write results
                    for prefix_list in results:
                        for prefix in prefix_list:
                            out_file.write(prefix + "\n")
                        total_prefixes += len(prefix_list)
                    
                    # Update progress
                    pbar.update(len(chunk))
                    chunk = []
                    
                    # Show current stats
                    pbar.set_postfix(prefixes=f"{total_prefixes:,}")
            
            # Process remaining lines
            if chunk:
                sub_chunks = [chunk[i:i+max(1, len(chunk)//NUM_WORKERS)] for i in range(0, len(chunk), max(1, len(chunk)//NUM_WORKERS))]
                
                with mp.Pool(NUM_WORKERS) as pool:
                    results = pool.map(process_chunk, sub_chunks)
                
                for prefix_list in results:
                    for prefix in prefix_list:
                        out_file.write(prefix + "\n")
                    total_prefixes += len(prefix_list)
                
                pbar.update(len(chunk))
            
            pbar.close()
    
    print(f"\n✅ Generated {total_prefixes:,} prefix samples from {processed_lines:,} sentences")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    
    # Print file size info
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"📏 Output file size: {file_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sentence prefixes from Wikipedia sentences")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input file with sentences")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file for prefixes")
    parser.add_argument("--min-len", type=int, default=MIN_LEN, help="Minimum tokens in prefix")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="Maximum tokens in prefix")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for processing")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    MIN_LEN = args.min_len
    MAX_LEN = args.max_len
    CHUNK_SIZE = args.chunk_size
    NUM_WORKERS = args.workers
    
    main()
