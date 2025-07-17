import re
import os
import random
from datasets import load_dataset, get_dataset_config_info
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

    selected_snapshot = None
    # Check available configs dynamically for a more robust approach
    try:
        config_info = get_dataset_config_info("wikimedia/wikipedia")
        available_configs = list(config_info.keys())
        print(f"Checking for available snapshots for 'wikimedia/wikipedia'. Total available: {len(available_configs)}")

        for snapshot_candidate in preferred_snapshots:
            if snapshot_candidate in available_configs:
                selected_snapshot = snapshot_candidate
                print(f"Found and selected snapshot: {selected_snapshot}")
                break
        
        if selected_snapshot is None:
            # Fallback if preferred snapshots not found, try to find *any* English snapshot
            print("Preferred English snapshots not found. Searching for any available English snapshot...")
            for config in available_configs:
                if config.endswith(".en"):
                    selected_snapshot = config
                    print(f"Found and selected fallback English snapshot: {selected_snapshot}")
                    break
        
        if selected_snapshot is None:
            raise ValueError(f"No English snapshot found for 'wikimedia/wikipedia' dataset. Available configs: {available_configs}")

    except Exception as e:
        print(f"Error checking dataset configurations: {e}")
        print("Please ensure 'datasets' library is up to date and you have internet access.")
        print("Falling back to a hardcoded snapshot if dynamic check fails or no internet.")
        selected_snapshot = "20231101.en" # Fallback to a known good one from the error message

    try:
        # Load the selected Wikipedia snapshot
        dataset = load_dataset("wikimedia/wikipedia", selected_snapshot, split="train", streaming=True)

        print(f"Loading wikimedia/wikipedia dataset ('{selected_snapshot}' snapshot) in streaming mode...")
        with open(output_filename, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc="Processing Wikipedia articles"):
                page_text = example['text']

                cleaned_text = clean_wikipedia_text(page_text)
                if random.randint(0, 100) < 10:
                    continue

                if len(cleaned_text) < 100:
                    continue

                f.write(cleaned_text + "\n\n")
                current_size = os.path.getsize(output_filename)
                num_pages += 1

                if current_size >= TARGET_SIZE_BYTES:
                    print(f"\nTarget size reached: {current_size / (1024**3):.2f} GB.")
                    print(f"Total clean articles processed: {num_pages}")
                    break

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
