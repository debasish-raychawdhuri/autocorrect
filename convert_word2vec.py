#!/usr/bin/env python3
"""
Convert word2vec binary format to custom numpy format
Run this in a venv with older Python/Cython where gensim works
"""
import numpy as np
import argparse
from gensim.models import KeyedVectors
import json
import pickle
from tqdm import tqdm

def convert_word2vec_to_numpy(input_path, output_path):
    """Convert word2vec binary format to numpy arrays"""
    print(f"Loading word2vec model from {input_path}")
    
    # Load the word2vec model
    model = KeyedVectors.load_word2vec_format(input_path, binary=True)
    
    print(f"Model loaded with {len(model.key_to_index)} words, {model.vector_size} dimensions")
    
    # Extract vocabulary and vectors
    words = list(model.key_to_index.keys())
    print("Converting vectors to numpy arrays...")
    vectors = np.array([model[word] for word in tqdm(words, desc="Extracting vectors")], dtype=np.float32)
    
    # Create word-to-index mapping
    word_to_idx = {word: i for i, word in enumerate(words)}
    
    # Save as numpy format
    data = {
        'words': words,
        'vectors': vectors,
        'word_to_idx': word_to_idx,
        'vector_size': model.vector_size,
        'vocab_size': len(words)
    }
    
    print(f"Saving to {output_path} (this may take several minutes for large files...)")
    np.savez_compressed(output_path, **data)
    print("Saving complete!")
    
    print(f"Conversion complete!")
    print(f"Vocabulary size: {len(words)}")
    print(f"Vector dimension: {model.vector_size}")
    # Get file size
    import os
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Output file size: {file_size_mb:.2f} MB")

def convert_word2vec_to_pickle(input_path, output_path):
    """Convert word2vec binary format to pickle format"""
    print(f"Loading word2vec model from {input_path}")
    
    # Load the word2vec model
    model = KeyedVectors.load_word2vec_format(input_path, binary=True)
    
    print(f"Model loaded with {len(model.key_to_index)} words, {model.vector_size} dimensions")
    
    # Extract vocabulary and vectors
    word_vectors = {}
    print("Converting vectors to dictionary format...")
    for word in tqdm(model.key_to_index.keys(), desc="Converting vectors"):
        word_vectors[word] = model[word].astype(np.float32)
    
    # Save as pickle
    data = {
        'word_vectors': word_vectors,
        'vector_size': model.vector_size,
        'vocab_size': len(word_vectors)
    }
    
    print(f"Saving to {output_path} (pickle format is faster than numpy...)") 
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saving complete!")
    
    print(f"Conversion complete!")
    print(f"Vocabulary size: {len(word_vectors)}")
    print(f"Vector dimension: {model.vector_size}")
    
    # Get file size
    import os
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Output file size: {file_size_mb:.2f} MB")

def convert_word2vec_to_json(input_path, output_path):
    """Convert word2vec binary format to JSON format (slower but more portable)"""
    print(f"Loading word2vec model from {input_path}")
    
    # Load the word2vec model
    model = KeyedVectors.load_word2vec_format(input_path, binary=True)
    
    print(f"Model loaded with {len(model.key_to_index)} words, {model.vector_size} dimensions")
    
    # Extract vocabulary and vectors
    word_vectors = {}
    for word in tqdm(model.key_to_index.keys(), desc="Converting vectors"):
        word_vectors[word] = model[word].astype(np.float32).tolist()
    
    # Save as JSON
    data = {
        'word_vectors': word_vectors,
        'vector_size': model.vector_size,
        'vocab_size': len(word_vectors)
    }
    
    print(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
    
    print(f"Conversion complete!")
    print(f"Vocabulary size: {len(word_vectors)}")
    print(f"Vector dimension: {model.vector_size}")

def main():
    parser = argparse.ArgumentParser(description="Convert word2vec binary format to custom formats")
    parser.add_argument("input", help="Input word2vec binary file")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--format", choices=["numpy", "pickle", "json"], default="numpy",
                       help="Output format (default: numpy)")
    args = parser.parse_args()
    
    if args.format == "numpy":
        convert_word2vec_to_numpy(args.input, args.output)
    elif args.format == "pickle":
        convert_word2vec_to_pickle(args.input, args.output)
    elif args.format == "json":
        convert_word2vec_to_json(args.input, args.output)

if __name__ == "__main__":
    main()