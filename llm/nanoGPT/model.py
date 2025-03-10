import torch 
import torch.nn as nn 
import math


class InputEmbedding(nn.Module): 
    def __init__(self, vocab_len, n_embedding):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_len, n_embedding)
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        # positional encoding for each token in the sequence has d_model dimensions.
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indicess
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class MaskedMultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_len: int = 512) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        # Linear transformation for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Linear transformation for the concatenated outputs
        self.w_o = nn.Linear(d_model, d_model)
        
        # register buffer for mask 
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))
        
    def mask_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dropout: nn.Dropout) -> torch.Tensor:
        # Q, K, V: (batch, n_heads, seq_len, head_dim)
        # mask: (batch, 1, seq_len, seq_len)
        # computer attention score : Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1]) # (batch, n_heads, seq_len, seq_len)
        
        # apply mask
        # scores: (batch, n_heads, seq_len, seq_len)
        mask_value = self.mask[:, :, :scores.size(2), :scores.size(3)] # (batch, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask_value == 0, float("-inf"))
        
        # Apply softmax to the last dimension
        attention = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attention = dropout(attention)
        
        # Multiply the attention scores by the value vectors
        # (batch, n_heads, seq_len, seq_len) * (batch, n_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)
        output = torch.matmul(attention, V) 
        return output, attention 
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.w_q(x) # (batch, seq_len, d_model) * (batch, d_model, d_model) -> (batch, seq_len, d_model)
        K = self.w_k(x)   # same 
        V = self.w_v(x) # same
        
        # Split the d_model (Q, k, V) dimension into n_heads
        # d_model = head_dim * n_heads
        
        batch_size = Q.shape[0]
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # apply attention
        x, self.attention_scores = self.mask_attention(Q, K, V, self.dropout)
        
        # combine all heads together 
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, d_model)
        # contiguous() -> make sure the tensor is stored in a contiguous chunk of memory
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        
        # apply projection
        x = self.w_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class LayerNormalization(nn.Module): 
        
    def __init__(self, d_model: int, eps: float = 1e-6) -> None: 
        super().__init__() 
        self.d_model = d_model 
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(d_model)) 
        self.beta = nn.Parameter(torch.zeros(d_model)) 
    
    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True) # get mean 
        std = x.std(dim=-1, keepdim=True)   # get varianceb 
        # normalize 
        x = (x - mean) / (std + self.eps)
        # scale and shift: y = gamma * x + beta
        # gamma for scaling, beta for shifting
        y = self.gamma * x + self.beta
        return y


class Block(nn.Module): 
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None: 
        super().__init__() 
        self.attn = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # norm + residual -> attention -> norm + residual -> feedforward -> norm + residual
        x = x + self.attn(self.norm1(x)) 
        x = x + self.ffwd(self.norm2(x))
        return x
    
class nanoGPT(nn.Module): 
    def __init__(self, vocab_len, d_model, n_layers, n_heads, d_ff, dropout, max_len = 1024) -> None: 
        super().__init__() 
        self.embedding = InputEmbedding(vocab_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers: 
            x = layer(x)
        x = self.fc(x)
        return x
    
def init_model(
    model_name = "nanoGPT",
    d_model = 512,
    n_layers = 3,
    n_heads = 4,
    dropout = 0.15,
    vocab_len = 256
):
    # hyperparameters
    d_ff = 4*d_model     # dimension of feedforward network | 4 times d_model
    nano_gpt = nanoGPT(vocab_len, d_model, n_layers, n_heads, d_ff, dropout)
    return nano_gpt
