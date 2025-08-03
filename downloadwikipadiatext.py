import re
import os
import random
from datasets import load_dataset
from tqdm.auto import tqdm

# ... (clean_wikipedia_text function remains the same as before) ...
def clean_wikipedia_text(text):
    """
    Cleans a block of text from Wikipedia by removing MediaWiki markup
    like math tags, image/file tags, internal links (keeping display text),
    templates, reference brackets, and normalizing whitespace.
    """
    # Regular expressions for cleaning
    MATH_TAG_PATTERN = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL)
    IMAGE_FILE_PATTERN = re.compile(r'\[\[(?:File|Image|Media):[^\]]*\]\]')
    INTERNAL_LINK_PATTERN = re.compile(r'\[\[(?:[^:\]\|]*:)?([^\]\|]+)(?:\|([^\]]+))?\]\]')
    TEMPLATE_PATTERN = re.compile(r'\{\{[^}]*\}\}')
    REFERENCE_BRACKETS_PATTERN = re.compile(r'\[\d+\]|\[[a-zA-Z]+\]')
    HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    text = MATH_TAG_PATTERN.sub('', text)

    def replace_link(match):
        display_text = match.group(2)
        if display_text:
            return display_text
        else:
            article_name = match.group(1)
            if ':' in article_name and not article_name.startswith('http'):
                 article_name = article_name.split(':', 1)[1]
            return article_name
    text = INTERNAL_LINK_PATTERN.sub(replace_link, text)

    text = IMAGE_FILE_PATTERN.sub('', text)
    text = TEMPLATE_PATTERN.sub('', text)
    text = REFERENCE_BRACKETS_PATTERN.sub('', text)
    text = HTML_TAG_PATTERN.sub('', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()

    return text
# ... (end of clean_wikipedia_text function) ...


def download_and_clean_wikipedia(target_gb=5, output_filename="clean_wikipedia_data.txt"):
    """
    Downloads and cleans Wikipedia text data from the wikimedia/wikipedia dataset,
    writing it to a file until a specified target size is reached.

    Args:
        target_gb (int): The target size of the output file in gigabytes.
        output_filename (str): The name of the output text file.
    """
    TARGET_SIZE_BYTES = target_gb * 1024 * 1024 * 1024
    current_size = 0
    num_pages = 0

    print(f"Starting data extraction and cleaning from Wikipedia (targeting {target_gb} GB)...")

    # Define preferred snapshots in descending order of recency
    # Based on the error message, '20231101.en' is currently available.
    # We can add a more recent one for future proofing if it becomes available.
    preferred_snapshots = ["20240301.en", "20231101.en", "20230901.en"] # Add other plausible recent dates if known

    # Try snapshots in order until one works
    selected_snapshot = None
    for snapshot_candidate in preferred_snapshots:
        try:
            # Test if this snapshot works by trying to load the dataset builder
            from datasets import load_dataset_builder
            load_dataset_builder("wikimedia/wikipedia", snapshot_candidate)
            selected_snapshot = snapshot_candidate
            print(f"Found and selected snapshot: {selected_snapshot}")
            break
        except:
            continue
    
    if selected_snapshot is None:
        print("All preferred snapshots failed. Using fallback snapshot.")
        selected_snapshot = "20231101.en"

    try:
        # Download and process Wikipedia XML dumps directly
        import urllib.request
        import bz2
        import xml.etree.ElementTree as ET
        
        # Try multiple dump files until we reach target size
        dump_files = [
            "enwiki-latest-pages-articles1.xml-p1p41242.bz2",
            "enwiki-latest-pages-articles2.xml-p41243p151573.bz2",
            "enwiki-latest-pages-articles3.xml-p151574p311329.bz2"
        ]
        
        print(f"Downloading and processing Wikipedia dumps directly...")
        
        with open(output_filename, "w", encoding="utf-8") as f:
            for dump_file in dump_files:
                if os.path.getsize(output_filename) >= TARGET_SIZE_BYTES:
                    break
                    
                print(f"Downloading {dump_file}...")
                urllib.request.urlretrieve(f"https://dumps.wikimedia.org/enwiki/latest/{dump_file}", dump_file)
                
                print(f"Processing {dump_file}...")
                with bz2.open(dump_file, 'rt', encoding='utf-8') as xml_file:
                    current_text = ""
                    in_text = False
                    
                    for line in tqdm(xml_file, desc=f"Processing {dump_file}"):
                        if '<text' in line and 'xml:space="preserve"' in line:
                            in_text = True
                            current_text = line
                        elif in_text:
                            current_text += line
                            if '</text>' in line:
                                in_text = False
                                # Extract and clean text
                                text_match = re.search(r'<text[^>]*>(.*?)</text>', current_text, re.DOTALL)
                                if text_match:
                                    raw_text = text_match.group(1)
                                    cleaned_text = clean_wikipedia_text(raw_text)
                                    
                                    if random.randint(0, 100) < 10:
                                        continue
                                    
                                    if len(cleaned_text) < 100:
                                        continue
                                    
                                    f.write(cleaned_text + "\n\n")
                                    num_pages += 1
                                    
                                    current_size = os.path.getsize(output_filename)
                                    if current_size >= TARGET_SIZE_BYTES:
                                        print(f"\nTarget size reached: {current_size / (1024**3):.2f} GB.")
                                        print(f"Total clean articles processed: {num_pages}")
                                        return
                                current_text = ""
                
                # Clean up downloaded file
                os.remove(dump_file)

        print(f"Final data size: {os.path.getsize(output_filename) / (1024**3):.2f} GB in {output_filename}")
        print("Cleaning complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an active internet connection and the 'datasets' library is installed (`pip install datasets`).")
        print("Also, check for sufficient disk space. If the 'wikimedia/wikipedia' dataset fails,")
        print("verify its exact version and sub-config name on Hugging Face Hub:")
        print("https://huggingface.co/datasets/wikimedia/wikipedia")


if __name__ == "__main__":
    download_and_clean_wikipedia(target_gb=1, output_filename="clean_wikipedia_for_autocorrect.txt")
