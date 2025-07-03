import os
import re
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# -------- CONFIG --------
INPUT_FILE = "./prefix_samples.txt"
OUTPUT_FILE = "./error_samples.jsonl"
ERRORS_PER_SAMPLE = 10
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
NUM_WORKERS = cpu_count()  # or set manually
# ------------------------

def smart_tokenize(text):
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

def random_corrupt(word):
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
    tokens = smart_tokenize(line)
    if len(tokens) < 2:
        return []

    last_word = tokens[-1]
    prefix = tokens[:-1]

    samples = []
    used = set()

    while len(samples) < ERRORS_PER_SAMPLE:
        corrupted = random_corrupt(last_word)
        if corrupted != last_word and corrupted not in used:
            used.add(corrupted)
            samples.append({
                "input": " ".join(prefix + [corrupted]),
                "target": last_word
            })

    # Add clean version
    samples.append({
        "input": line,
        "target": last_word
    })

    return samples

def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with Pool(NUM_WORKERS) as pool, open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for all_sample_lists in tqdm(pool.imap_unordered(generate_error_samples, lines, chunksize=128), total=len(lines)):
            for sample in all_sample_lists:
                out.write(json.dumps(sample) + "\n")

    print(f"✅ Multiprocessing complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

