import json
import random
import re
import multiprocessing
from tqdm import tqdm
import math

def clean_word(word):
    """Keep only a-z, A-Z, and apostrophe. Lowercase for consistency."""
    return re.sub(r"[^a-zA-Z']", '', word).lower()

def pad_context(words, ctx_len=10):
    return [""] * max(0, ctx_len - len(words)) + words[-ctx_len:]

def random_edit(word, alphabet):
    if len(word) == 0:
        return word
    n_edits = random.randint(1, 3)
    word = list(word)
    for _ in range(n_edits):
        if not word:  # If word is empty after edits, skip further edits
            break
        op = random.choice(["insert", "delete", "replace"])
        idx = random.randint(0, len(word)-1)
        if op == "insert":
            word.insert(idx, random.choice(alphabet))
        elif op == "delete" and len(word) > 1:
            del word[idx]
        elif op == "replace":
            word[idx] = random.choice(alphabet)
    return ''.join(word)

def generate_samples_for_sentence(sentence, ctx_len=10, n_noisy=10, alphabet=None):
    if alphabet is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz'"
    # Clean and filter words
    words = [clean_word(w) for w in sentence.strip().split()]
    words = [w for w in words if w]  # Remove empty after cleaning
    results = []
    for i in range(1, len(words)):
        context = pad_context(words[:i], ctx_len)
        correct_word = words[i]
        if not correct_word:
            continue
        misspelled_versions = set()
        while len(misspelled_versions) < n_noisy:
            misspelled = random_edit(correct_word, alphabet)
            if misspelled != correct_word and misspelled:
                misspelled_versions.add(misspelled)
        misspelled_versions = list(misspelled_versions)
        misspelled_versions.append(correct_word)
        for misspelled in misspelled_versions:
            for k in range(len(correct_word) + 1):
                prefix = correct_word[:k]
                next_char = correct_word[k] if k < len(correct_word) else "<eow>"
                results.append({
                    "context": context,
                    "misspelled": misspelled,
                    "generated_prefix": prefix,
                    "next_char": next_char
                })
    return results

def worker(args):
    chunk, ctx_len, n_noisy, alphabet = args
    batch = []
    for sentence in chunk:
        batch.extend(generate_samples_for_sentence(sentence, ctx_len, n_noisy, alphabet))
    return batch

def process_sentences_parallel(in_file, out_file, ctx_len=10, n_noisy=10, num_workers=None, batch_size=1000):
    # Count total sentences for progress tracking
    with open(in_file, encoding="utf-8") as fin:
        total_sentences = sum(1 for line in fin if line.strip())
    
    num_workers = num_workers or multiprocessing.cpu_count()
    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    
    # Open output file for writing
    with open(out_file, "w", encoding="utf-8") as fout:
        with open(in_file, encoding="utf-8") as fin:
            sentences_batch = []
            processed_sentences = 0
            
            with tqdm(total=total_sentences, desc="Processing sentences") as pbar:
                for line in fin:
                    sentence = line.strip()
                    if sentence:
                        sentences_batch.append(sentence)
                        
                        # Process batch when it reaches batch_size
                        if len(sentences_batch) >= batch_size:
                            # Create chunks for multiprocessing
                            chunk_size = math.ceil(len(sentences_batch) / num_workers)
                            chunks = [sentences_batch[i:i+chunk_size] for i in range(0, len(sentences_batch), chunk_size)]
                            args_list = [(chunk, ctx_len, n_noisy, alphabet) for chunk in chunks]
                            
                            # Process in parallel and write immediately
                            with multiprocessing.Pool(processes=num_workers) as pool:
                                all_results = pool.map(worker, args_list)
                                for batch_results in all_results:
                                    for sample in batch_results:
                                        fout.write(json.dumps(sample) + "\n")
                            
                            processed_sentences += len(sentences_batch)
                            pbar.update(len(sentences_batch))
                            sentences_batch = []  # Clear batch from memory
                
                # Process remaining sentences
                if sentences_batch:
                    chunk_size = math.ceil(len(sentences_batch) / num_workers)
                    chunks = [sentences_batch[i:i+chunk_size] for i in range(0, len(sentences_batch), chunk_size)]
                    args_list = [(chunk, ctx_len, n_noisy, alphabet) for chunk in chunks]
                    
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        all_results = pool.map(worker, args_list)
                        for batch_results in all_results:
                            for sample in batch_results:
                                fout.write(json.dumps(sample) + "\n")
                    
                    processed_sentences += len(sentences_batch)
                    pbar.update(len(sentences_batch))

def process_sentences_to_multiple_files(in_file, out_dir, num_files=8, ctx_len=10, n_noisy=10, num_workers=None, batch_size=1000):
    """Process sentences and distribute output across multiple files"""
    import os
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Count total sentences for progress tracking
    with open(in_file, encoding="utf-8") as fin:
        total_sentences = sum(1 for line in fin if line.strip())
    
    num_workers = num_workers or multiprocessing.cpu_count()
    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    
    # Open multiple output files
    output_files = []
    file_handles = []
    for i in range(num_files):
        filename = os.path.join(out_dir, f"training_data_{i:03d}.json")
        output_files.append(filename)
        file_handles.append(open(filename, "w", encoding="utf-8"))
    
    try:
        current_file_idx = 0
        sample_counts = [0] * num_files
        
        with open(in_file, encoding="utf-8") as fin:
            sentences_batch = []
            processed_sentences = 0
            
            with tqdm(total=total_sentences, desc="Processing sentences") as pbar:
                for line in fin:
                    sentence = line.strip()
                    if sentence:
                        sentences_batch.append(sentence)
                        
                        # Process batch when it reaches batch_size
                        if len(sentences_batch) >= batch_size:
                            # Create chunks for multiprocessing
                            chunk_size = math.ceil(len(sentences_batch) / num_workers)
                            chunks = [sentences_batch[i:i+chunk_size] for i in range(0, len(sentences_batch), chunk_size)]
                            args_list = [(chunk, ctx_len, n_noisy, alphabet) for chunk in chunks]
                            
                            # Process in parallel and distribute across files
                            with multiprocessing.Pool(processes=num_workers) as pool:
                                all_results = pool.map(worker, args_list)
                                for batch_results in all_results:
                                    for sample in batch_results:
                                        file_handles[current_file_idx].write(json.dumps(sample) + "\n")
                                        sample_counts[current_file_idx] += 1
                                        current_file_idx = (current_file_idx + 1) % num_files
                            
                            processed_sentences += len(sentences_batch)
                            pbar.update(len(sentences_batch))
                            sentences_batch = []  # Clear batch from memory
                
                # Process remaining sentences
                if sentences_batch:
                    chunk_size = math.ceil(len(sentences_batch) / num_workers)
                    chunks = [sentences_batch[i:i+chunk_size] for i in range(0, len(sentences_batch), chunk_size)]
                    args_list = [(chunk, ctx_len, n_noisy, alphabet) for chunk in chunks]
                    
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        all_results = pool.map(worker, args_list)
                        for batch_results in all_results:
                            for sample in batch_results:
                                file_handles[current_file_idx].write(json.dumps(sample) + "\n")
                                sample_counts[current_file_idx] += 1
                                current_file_idx = (current_file_idx + 1) % num_files
                    
                    processed_sentences += len(sentences_batch)
                    pbar.update(len(sentences_batch))
    
    finally:
        # Close all file handles
        for fh in file_handles:
            fh.close()
    
    # Print summary and create metadata
    total_samples = sum(sample_counts)
    total_size = 0
    metadata = {
        "total_samples": total_samples,
        "total_files": num_files,
        "files": []
    }
    
    print(f"\nOutput files created in {out_dir}:")
    for i, (filename, count) in enumerate(zip(output_files, sample_counts)):
        size = os.path.getsize(filename)
        total_size += size
        
        # Add to metadata
        metadata["files"].append({
            "filename": f"training_data_{i:03d}.json",
            "samples": count,
            "size_bytes": size
        })
        
        print(f"  training_data_{i:03d}.json: {count:,} samples, {size / (1024**2):.1f} MB")
    
    metadata["total_size_bytes"] = total_size
    
    # Save metadata file
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    
    print(f"\nTotal: {total_samples:,} samples across {num_files} files, {total_size / (1024**2):.1f} MB")
    print(f"📊 Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="sentences.txt")
    parser.add_argument("--outfile", type=str, default="autogen_char_data.json")
    parser.add_argument("--outdir", type=str, default="training_data", help="Output directory for multiple files")
    parser.add_argument("--dataloader_workers", type=int, default=8, help="Number of DataLoader workers (determines number of output files)")
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--n_noisy", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None, help="Number of processes for generation (defaults to all cores)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of sentences to process before writing to disk")
    parser.add_argument("--multi_files", action="store_true", help="Generate multiple files for DataLoader workers")
    args = parser.parse_args()
    
    if args.multi_files:
        # Create as many files as there will be DataLoader workers
        process_sentences_to_multiple_files(args.infile, args.outdir, args.dataloader_workers, ctx_len=args.ctx_len, n_noisy=args.n_noisy, num_workers=args.workers, batch_size=args.batch_size)
    else:
        process_sentences_parallel(args.infile, args.outfile, ctx_len=args.ctx_len, n_noisy=args.n_noisy, num_workers=args.workers, batch_size=args.batch_size)

