import os
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# -------- CONFIG --------
WIKI_DIR = "./wiki"
OUTPUT_FILE = "./wiki_sentences.txt"
MIN_LEN = 20     # minimum sentence length (characters)
MAX_LEN = 300    # maximum sentence length (characters)
BATCH_SIZE = 10000  # process sentences in batches to manage memory
# -------------------------

def extract_sentences_from_text(text):
    """Extract sentences from text using NLTK sentence tokenizer"""
    sentences = sent_tokenize(text)
    sentences = [s.strip().replace("\n", " ").replace("\r", " ") for s in sentences]
    # Filter by length and remove sentences with too many special characters
    filtered_sentences = []
    for s in sentences:
        if MIN_LEN <= len(s) <= MAX_LEN:
            # Basic quality filter: sentence should be mostly alphabetic
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / len(s)
            if alpha_ratio > 0.7:  # At least 70% alphabetic characters + spaces
                filtered_sentences.append(s)
    return filtered_sentences

def process_large_file(file_path, output_file):
    """Process large wiki file in chunks to manage memory"""
    print(f"📖 Processing: {file_path}")
    
    sentence_count = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            text_buffer = ""
            
            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break
                
                text_buffer += chunk
                
                # Process complete paragraphs (split on double newlines)
                paragraphs = text_buffer.split('\n\n')
                
                # Keep the last incomplete paragraph in buffer
                text_buffer = paragraphs[-1]
                
                # Process complete paragraphs
                for paragraph in paragraphs[:-1]:
                    if paragraph.strip():
                        sentences = extract_sentences_from_text(paragraph)
                        for sentence in sentences:
                            outfile.write(sentence + '\n')
                            sentence_count += 1
                            
                            if sentence_count % BATCH_SIZE == 0:
                                print(f"  ✅ Processed {sentence_count:,} sentences...")
            
            # Process remaining text in buffer
            if text_buffer.strip():
                sentences = extract_sentences_from_text(text_buffer)
                for sentence in sentences:
                    outfile.write(sentence + '\n')
                    sentence_count += 1
    
    return sentence_count

def main():
    print("🚀 Starting Wikipedia sentence extraction...")
    
    total_sentences = 0
    
    # Process all text files in wiki directory
    for fname in os.listdir(WIKI_DIR):
        if fname.endswith(".txt"):
            file_path = os.path.join(WIKI_DIR, fname)
            
            # For the first file, create new output file; for subsequent files, append
            mode = 'w' if total_sentences == 0 else 'a'
            temp_output = OUTPUT_FILE + '.tmp'
            
            if total_sentences == 0:
                sentence_count = process_large_file(file_path, OUTPUT_FILE)
            else:
                # Append to existing file
                sentence_count = process_large_file(file_path, temp_output)
                # Append temp file to main output file
                with open(temp_output, 'r', encoding='utf-8') as temp_file:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as main_file:
                        main_file.write(temp_file.read())
                os.remove(temp_output)
            
            total_sentences += sentence_count
            print(f"  📊 Extracted {sentence_count:,} sentences from {fname}")

    print(f"\n✅ Total extracted: {total_sentences:,} sentences from {WIKI_DIR}")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    
    # Print file size info
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"📏 Output file size: {file_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("📥 Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab')
    
    main()
