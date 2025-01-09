import os 
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
max_len = 512
batch_size=64


data = "" 
for file in DATA_DIR.glob("*.txt"):
    with open(file, "r") as f:
        data += f.read()


# create model predict on character level. 
vocab = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n """

# ignore characters that are not in the vocab
data = "".join([char for char in data if char in vocab])

vocab_len = len(vocab)
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}

encoder = lambda x: [char2idx[char] for char in x]
decoder = lambda x: "".join([idx2char[idx] for idx in x])

ratio_train = 0.9
data = torch.tensor(encoder(data), dtype=torch.long)

train_data = data[:int(len(data) * ratio_train)]
test_data = data[int(len(data) * ratio_train):]

# build dataset and dataloader. 

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]  # Shifted sequence
        return x, y

train_dataset = TextDataset(train_data, max_len)
test_dataset = TextDataset(test_data, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
