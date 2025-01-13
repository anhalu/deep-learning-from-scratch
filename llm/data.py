import os
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch
from tokenizer import BytePairEncoding

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / 'data'
    MAX_LEN = 256
    BATCH_SIZE = 400
    TRAIN_RATIO = 0.8
    NUM_MERGES = 2000
    VOCAB = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n """

class TextDataset(Dataset):
    """Dataset for handling text data with sliding window sequences"""
    
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx + self.seq_len > len(self.data):
            idx = len(self.data) - self.seq_len - 1
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

class DataProcessor:
    """Handles data loading, processing and tokenization"""
    
    def __init__(self, tokenizer: BytePairEncoding | None = None):
        self.tokenizer = tokenizer
        self.vocab_size = 0
        self.train_data = None
        self.test_data = None

    def load_data(self) -> str:
        """Load and combine all text files from data directory"""
        data = ""
        for file in Config.DATA_DIR.glob("*.txt"):
            with open(file, "r", encoding='utf-8') as f:
                data += f.read()
        return "".join([char for char in data if char in Config.VOCAB])


    def process_data(self) -> Tuple[TextDataset, TextDataset]:
        """Process data and create train/test datasets"""
        data = self.load_data()
        
        encoded_data = torch.tensor(self.tokenizer.encode(data), dtype=torch.long)
        split_idx = int(len(encoded_data) * Config.TRAIN_RATIO)
        
        self.train_data = TextDataset(encoded_data[:split_idx], Config.MAX_LEN)
        self.test_data = TextDataset(encoded_data[split_idx:], Config.MAX_LEN)
        
        return self.train_data, self.test_data

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoader instances for train and test data"""
        if not (self.train_data and self.test_data):
            self.process_data()
            
        train_loader = DataLoader(
            self.train_data, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_data, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False
        )
        return train_loader, test_loader

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)