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

def process_sentences_parallel(in_file, out_file, ctx_len=10, n_noisy=10, num_workers=None):
    with open(in_file, encoding="utf-8") as fin:
        sentences = [line.strip() for line in fin if line.strip()]
    total = len(sentences)
    num_workers = num_workers or multiprocessing.cpu_count()
    chunk_size = math.ceil(total / num_workers)
    chunks = [sentences[i:i+chunk_size] for i in range(0, total, chunk_size)]
    alphabet = "abcdefghijklmnopqrstuvwxyz'"

    args_list = [(chunk, ctx_len, n_noisy, alphabet) for chunk in chunks]

    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch in tqdm(pool.imap_unordered(worker, args_list), total=len(chunks), desc="Processing"):
            results.extend(batch)

    with open(out_file, "w", encoding="utf-8") as fout:
        for sample in tqdm(results, desc="Writing"):
            fout.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="sentences.txt")
    parser.add_argument("--outfile", type=str, default="autogen_char_data.json")
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--n_noisy", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None, help="Number of processes (defaults to all cores)")
    args = parser.parse_args()
    process_sentences_parallel(args.infile, args.outfile, ctx_len=args.ctx_len, n_noisy=args.n_noisy, num_workers=args.workers)

