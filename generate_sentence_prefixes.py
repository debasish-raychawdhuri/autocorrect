import os
import re

# -------- CONFIG --------
INPUT_FILE = "./sentences.txt"
OUTPUT_FILE = "./prefix_samples.txt"
MIN_LEN = 2     # minimum number of tokens to include a prefix
MAX_LEN = 100   # max number of tokens per prefix (optional)
# ------------------------

def smart_tokenize(text):
    """
    Tokenize a sentence into words and punctuation.
    Keeps contractions like "don't" together.
    """
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

def generate_left_contexts(sentence):
    tokens = smart_tokenize(sentence)
    prefixes = [" ".join(tokens[:i+1]) for i in range(len(tokens))]
    # Optional: filter out very short or long
    return [p for p in prefixes if len(p.split()) >= MIN_LEN and len(p.split()) <= MAX_LEN]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    all_prefixes = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prefixes = generate_left_contexts(line)
            all_prefixes.extend(prefixes)

    print(f"✅ Generated {len(all_prefixes):,} prefix samples.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for prefix in all_prefixes:
            out.write(prefix + "\n")

    print(f"📁 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

