import os
import requests
import re
from tqdm import tqdm

# -------- CONFIG --------
TOP_N = 10
SAVE_DIR = "./gutenberg_books"

# Manually curated top download IDs from Project Gutenberg
TOP_BOOK_IDS = [
    1342,   # Pride and Prejudice
    84,     # Frankenstein
    2701,   # Moby Dick
    11,     # Alice in Wonderland
    76,     # Huckleberry Finn
    1661,   # Sherlock Holmes
    98,     # A Tale of Two Cities
    2542,   # A Doll’s House
    1952,   # The Yellow Wallpaper
    2600    # War and Peace
][:TOP_N]
# -------------------------

def strip_gutenberg_boilerplate(text):
    """Remove Gutenberg header/footer markers."""
    start = re.search(r"\*\*\* START OF (.*?) \*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* END OF (.*?) \*\*\*", text, re.IGNORECASE)
    if start and end:
        return text[start.end():end.start()].strip()
    return text.strip()

def try_gutenberg_urls(book_id):
    """Try multiple known URL patterns for Gutenberg .txt files."""
    base = f"https://www.gutenberg.org/files/{book_id}"
    candidates = [
        f"{base}/{book_id}.txt",
        f"{base}/{book_id}.txt.utf-8",
        f"{base}/{book_id}-0.txt",
        f"{base}/{book_id}-0.txt.utf-8",
        f"{base}/{book_id}-h/{book_id}-h.txt",
        f"{base}/{book_id}-h/{book_id}-h.htm"
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and len(r.text.strip()) > 1000:
                return r.text, url
        except Exception:
            continue
    raise RuntimeError(f"No working URL found for book #{book_id}")

def download_and_clean(book_id, save_dir):
    try:
        raw, url = try_gutenberg_urls(book_id)
        cleaned = strip_gutenberg_boilerplate(raw)
        filename = os.path.join(save_dir, f"book_{book_id}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"✅ Book #{book_id} downloaded from {url}")
    except Exception as e:
        print(f"❌ Book #{book_id} failed: {e}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for book_id in tqdm(TOP_BOOK_IDS, desc="Downloading books"):
        download_and_clean(book_id, SAVE_DIR)

if __name__ == "__main__":
    main()

