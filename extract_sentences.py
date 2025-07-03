import os
import nltk
from nltk.tokenize import sent_tokenize

# -------- CONFIG --------
BOOKS_DIR = "./gutenberg_books"
OUTPUT_FILE = "./sentences.txt"
MIN_LEN = 20     # minimum sentence length (characters)
MAX_LEN = 300    # maximum sentence length (characters)
# -------------------------

def extract_sentences_from_text(text):
    sentences = sent_tokenize(text)
    sentences = [s.strip().replace("\n", " ") for s in sentences]
    return [s for s in sentences if MIN_LEN <= len(s) <= MAX_LEN]

def main():
    all_sentences = []
    for fname in os.listdir(BOOKS_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(BOOKS_DIR, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()
                sentences = extract_sentences_from_text(text)
                all_sentences.extend(sentences)

    print(f"✅ Extracted {len(all_sentences):,} sentences from {BOOKS_DIR}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for sentence in all_sentences:
            out.write(sentence + "\n")
    print(f"📁 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    nltk.download('punkt_tab')
    main()

