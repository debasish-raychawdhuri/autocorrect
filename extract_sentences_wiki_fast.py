import os
import re
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# -------- CONFIG --------
WIKI_DIR = "./wiki"
OUTPUT_FILE = "./wiki_sentences_fast.txt"
MIN_LEN = 20     # minimum sentence length (characters)
MAX_LEN = 300    # maximum sentence length (characters)
CHUNK_SIZE = 1024 * 1024 * 10  # 10MB chunks for processing
NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
# -------------------------

def simple_sentence_split(text):
    """Fast sentence splitting using regex (alternative to NLTK for speed)"""
    # Split on sentence endings followed by whitespace and capital letter
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Clean whitespace and newlines
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        
        # Filter by length
        if MIN_LEN <= len(sentence) <= MAX_LEN:
            # Basic quality filter: mostly alphabetic content
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in sentence) / len(sentence)
            if alpha_ratio > 0.7:
                cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def process_text_chunk(text_chunk):
    """Process a chunk of text and return extracted sentences"""
    sentences = simple_sentence_split(text_chunk)
    return sentences

def process_file_parallel(file_path, output_file, num_workers=NUM_WORKERS):
    """Process large file using multiprocessing for speed"""
    print(f"📖 Processing: {file_path}")
    print(f"🔧 Using {num_workers} worker processes")
    
    file_size = os.path.getsize(file_path)
    print(f"📏 File size: {file_size / (1024*1024*1024):.2f} GB")
    
    total_sentences = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Create progress bar
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing")
            
            text_buffer = ""
            chunks_to_process = []
            
            while True:
                chunk = infile.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                pbar.update(len(chunk.encode('utf-8')))
                text_buffer += chunk
                
                # Split on paragraph boundaries to avoid cutting sentences
                paragraphs = text_buffer.split('\n\n')
                text_buffer = paragraphs[-1]  # Keep incomplete paragraph
                
                # Collect complete paragraphs for processing
                for paragraph in paragraphs[:-1]:
                    if paragraph.strip():
                        chunks_to_process.append(paragraph)
                
                # Process chunks in batches to manage memory
                if len(chunks_to_process) >= num_workers * 2:
                    with mp.Pool(num_workers) as pool:
                        results = pool.map(process_text_chunk, chunks_to_process)
                    
                    # Write results
                    for sentences in results:
                        for sentence in sentences:
                            outfile.write(sentence + '\n')
                            total_sentences += 1
                    
                    chunks_to_process = []
                    
                    if total_sentences % 50000 == 0:
                        pbar.set_postfix(sentences=f"{total_sentences:,}")
            
            # Process remaining chunks
            if chunks_to_process:
                with mp.Pool(num_workers) as pool:
                    results = pool.map(process_text_chunk, chunks_to_process)
                
                for sentences in results:
                    for sentence in sentences:
                        outfile.write(sentence + '\n')
                        total_sentences += 1
            
            # Process remaining buffer
            if text_buffer.strip():
                sentences = process_text_chunk(text_buffer)
                for sentence in sentences:
                    outfile.write(sentence + '\n')
                    total_sentences += 1
            
            pbar.close()
    
    return total_sentences

def main():
    print("🚀 Starting fast Wikipedia sentence extraction...")
    print(f"⚙️  Configuration:")
    print(f"   - Min sentence length: {MIN_LEN} chars")
    print(f"   - Max sentence length: {MAX_LEN} chars")
    print(f"   - Chunk size: {CHUNK_SIZE / (1024*1024):.1f} MB")
    print(f"   - Workers: {NUM_WORKERS}")
    
    total_sentences = 0
    
    # Process all text files in wiki directory
    wiki_files = [f for f in os.listdir(WIKI_DIR) if f.endswith('.txt')]
    
    if not wiki_files:
        print(f"❌ No .txt files found in {WIKI_DIR}")
        return
    
    for i, fname in enumerate(wiki_files):
        file_path = os.path.join(WIKI_DIR, fname)
        print(f"\n📂 Processing file {i+1}/{len(wiki_files)}: {fname}")
        
        # For first file, create new; for others, append
        if i == 0:
            sentence_count = process_file_parallel(file_path, OUTPUT_FILE)
        else:
            temp_output = OUTPUT_FILE + f'.tmp_{i}'
            sentence_count = process_file_parallel(file_path, temp_output)
            
            # Append to main file
            with open(temp_output, 'r', encoding='utf-8') as temp_file:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as main_file:
                    main_file.write(temp_file.read())
            os.remove(temp_output)
        
        total_sentences += sentence_count
        print(f"  ✅ Extracted {sentence_count:,} sentences from {fname}")

    print(f"\n🎉 Extraction complete!")
    print(f"📊 Total sentences: {total_sentences:,}")
    print(f"📁 Output file: {OUTPUT_FILE}")
    
    # Print output file info
    if os.path.exists(OUTPUT_FILE):
        output_size = os.path.getsize(OUTPUT_FILE)
        print(f"📏 Output size: {output_size / (1024*1024):.1f} MB")
        
        # Estimate compression ratio
        total_input_size = sum(os.path.getsize(os.path.join(WIKI_DIR, f)) 
                              for f in wiki_files)
        compression_ratio = output_size / total_input_size * 100
        print(f"📉 Size reduction: {100-compression_ratio:.1f}% (from {total_input_size/(1024*1024*1024):.2f} GB)")

if __name__ == "__main__":
    main()
