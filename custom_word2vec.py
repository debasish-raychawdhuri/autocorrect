#!/usr/bin/env python3
"""
Custom word2vec loader that doesn't require gensim
Works with CUDA 12.8 environments
"""
import numpy as np
import pickle
import json
from typing import Dict, List, Optional, Union

class CustomWord2Vec:
    """Custom word2vec loader that replaces gensim.KeyedVectors"""
    
    def __init__(self, word_vectors: Dict[str, np.ndarray], vector_size: int):
        self.word_vectors = word_vectors
        self.vector_size = vector_size
        self.vocab_size = len(word_vectors)
        
        # Create reverse mapping for compatibility
        self.key_to_index = {word: i for i, word in enumerate(word_vectors.keys())}
        self.index_to_key = {i: word for word, i in self.key_to_index.items()}
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary"""
        return word in self.word_vectors
    
    def __getitem__(self, word: str) -> np.ndarray:
        """Get vector for word"""
        if word not in self.word_vectors:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.word_vectors[word]
    
    def get_vector(self, word: str, norm: bool = False) -> np.ndarray:
        """Get vector for word with optional normalization"""
        if word not in self.word_vectors:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        vector = self.word_vectors[word]
        if norm:
            vector = vector / np.linalg.norm(vector)
        return vector
    
    def similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words"""
        if word1 not in self.word_vectors or word2 not in self.word_vectors:
            return 0.0
        
        vec1 = self.word_vectors[word1]
        vec2 = self.word_vectors[word2]
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        return np.dot(vec1_norm, vec2_norm)
    
    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Find most similar words"""
        if word not in self.word_vectors:
            return []
        
        target_vec = self.word_vectors[word]
        target_vec_norm = target_vec / np.linalg.norm(target_vec)
        
        similarities = []
        for other_word, other_vec in self.word_vectors.items():
            if other_word == word:
                continue
            
            other_vec_norm = other_vec / np.linalg.norm(other_vec)
            sim = np.dot(target_vec_norm, other_vec_norm)
            similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    @classmethod
    def load_from_numpy(cls, path: str) -> 'CustomWord2Vec':
        """Load from numpy format"""
        print(f"Loading custom word2vec from {path}")
        
        data = np.load(path, allow_pickle=True)
        words = data['words']
        vectors = data['vectors']
        vector_size = int(data['vector_size'])
        
        # Create word-to-vector mapping
        word_vectors = {word: vectors[i] for i, word in enumerate(words)}
        
        print(f"Loaded {len(word_vectors)} words, {vector_size} dimensions")
        return cls(word_vectors, vector_size)
    
    @classmethod
    def load_from_pickle(cls, path: str) -> 'CustomWord2Vec':
        """Load from pickle format"""
        print(f"Loading custom word2vec from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        word_vectors = data['word_vectors']
        vector_size = data['vector_size']
        
        print(f"Loaded {len(word_vectors)} words, {vector_size} dimensions")
        return cls(word_vectors, vector_size)
    
    @classmethod
    def load_from_json(cls, path: str) -> 'CustomWord2Vec':
        """Load from JSON format"""
        print(f"Loading custom word2vec from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        word_vectors = {word: np.array(vec, dtype=np.float32) 
                       for word, vec in data['word_vectors'].items()}
        vector_size = data['vector_size']
        
        print(f"Loaded {len(word_vectors)} words, {vector_size} dimensions")
        return cls(word_vectors, vector_size)

def load_custom_word2vec(path: str) -> CustomWord2Vec:
    """
    Load custom word2vec format automatically detecting the format
    """
    if path.endswith('.npz'):
        return CustomWord2Vec.load_from_numpy(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        return CustomWord2Vec.load_from_pickle(path)
    elif path.endswith('.json'):
        return CustomWord2Vec.load_from_json(path)
    else:
        # Try to auto-detect format
        try:
            return CustomWord2Vec.load_from_numpy(path)
        except:
            try:
                return CustomWord2Vec.load_from_pickle(path)
            except:
                return CustomWord2Vec.load_from_json(path)

# Example usage and testing
if __name__ == "__main__":
    # Test the custom loader
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        # Load the model
        model = load_custom_word2vec(model_path)
        
        # Test basic functionality
        print(f"Model loaded: {model.vocab_size} words, {model.vector_size} dimensions")
        
        # Test some words
        test_words = ["the", "and", "is", "python", "computer"]
        for word in test_words:
            if word in model:
                print(f"'{word}' vector shape: {model[word].shape}")
                # Show similar words
                similar = model.most_similar(word, topn=3)
                print(f"  Similar to '{word}': {similar}")
            else:
                print(f"'{word}' not in vocabulary")
    else:
        print("Usage: python custom_word2vec.py <model_path>")
        print("Supported formats: .npz, .pkl, .json")