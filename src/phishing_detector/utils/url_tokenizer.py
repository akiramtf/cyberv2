"""Character-level tokenizer for URLs"""

import json
import numpy as np
from typing import List, Dict, Optional

class URLTokenizer:
    """
    Character-level tokenizer for URLs.
    Maps characters to integers for input into Deep Learning models.
    """
    
    def __init__(self, max_length: int = 200, vocab_size: int = None):
        self.max_length = max_length
        self.char_index = {}
        self.index_char = {}
        self.vocab_size = vocab_size
        
        # Reserved tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.char_index[self.pad_token] = 0
        self.char_index[self.unk_token] = 1
        self.index_char[0] = self.pad_token
        self.index_char[1] = self.unk_token
        
    def fit(self, urls: List[str]) -> None:
        """Build vocabulary from list of URLs"""
        unique_chars = set()
        for url in urls:
            unique_chars.update(list(url))
            
        # Sort for deterministic behavior
        sorted_chars = sorted(list(unique_chars))
        
        # Start index from 2 (0 and 1 are reserved)
        for i, char in enumerate(sorted_chars):
            idx = i + 2
            self.char_index[char] = idx
            self.index_char[idx] = char
            
        self.vocab_size = len(self.char_index)
        
    def transform(self, urls: List[str]) -> np.ndarray:
        """Convert URLs to padded integer sequences"""
        sequences = []
        
        for url in urls:
            seq = []
            for char in url:
                idx = self.char_index.get(char, self.char_index[self.unk_token])
                seq.append(idx)
            
            # Truncate if too long
            if len(seq) > self.max_length:
                seq = seq[:self.max_length]
            
            # Pad if too short
            padded_seq = seq + [self.char_index[self.pad_token]] * (self.max_length - len(seq))
            sequences.append(padded_seq)
            
        return np.array(sequences)
    
    def fit_transform(self, urls: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(urls)
        return self.transform(urls)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer vocabulary"""
        data = {
            "max_length": self.max_length,
            "char_index": self.char_index,
            "vocab_size": self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load(self, filepath: str) -> None:
        """Load tokenizer vocabulary"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.max_length = data["max_length"]
        self.char_index = data["char_index"]
        self.vocab_size = data["vocab_size"]
        # Reconstruct index_char
        self.index_char = {int(v): k for k, v in self.char_index.items()}
