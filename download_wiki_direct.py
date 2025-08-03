import re
import os
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm

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

def process_wikipedia_xml(xml_file, output_file, target_gb=1):
    """Process Wikipedia XML dump and extract clean text"""
    TARGET_SIZE_BYTES = target_gb * 1024 * 1024 * 1024
    
    print(f"Processing {xml_file}...")
    
    with bz2.open(xml_file, 'rt', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as out:
            current_element = ""
            in_text = False
            page_count = 0
            
            for line in tqdm(f, desc="Processing lines"):
                current_element += line
                
                if '<text' in line:
                    in_text = True
                elif '</text>' in line:
                    in_text = False
                    # Process the page
                    try:
                        # Extract text content
                        text_match = re.search(r'<text[^>]*>(.*?)</text>', current_element, re.DOTALL)
                        if text_match:
                            raw_text = text_match.group(1)
                            cleaned_text = clean_wikipedia_text(raw_text)
                            
                            if len(cleaned_text) > 100:  # Filter short articles
                                out.write(cleaned_text + "\n\n")
                                page_count += 1
                                
                                current_size = os.path.getsize(output_file)
                                if current_size >= TARGET_SIZE_BYTES:
                                    print(f"\nTarget size reached: {current_size / (1024**3):.2f} GB")
                                    print(f"Processed {page_count} articles")
                                    return
                    except Exception as e:
                        print(f"Error processing page: {e}")
                    
                    current_element = ""
                elif not in_text and '</page>' in line:
                    current_element = ""
    
    final_size = os.path.getsize(output_file)
    print(f"Final size: {final_size / (1024**3):.2f} GB")
    print(f"Processed {page_count} articles")

if __name__ == "__main__":
    process_wikipedia_xml('wiki_sample.xml.bz2', 'clean_wikipedia_for_autocorrect.txt', target_gb=1)