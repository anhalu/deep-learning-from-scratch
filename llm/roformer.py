import torch 
import torch.nn as nn 
import math
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputEmbedding(nn.Module): 
    def __init__(self, vocab_len, n_embedding):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_len, n_embedding)
    def forward(self, x):
        return self.embedding(x)


class RotaryPositionalEmbeddings(nn.Module): 
    def __init__(self, d_model: int, base: int = 10000) -> None: 
        super().__init__()
        self.base = base 
        self.d_model = d_model
        self.cos_cached = None
        self.sin_cached = None
        
    def _build_cached(self, x: torch.Tensor) -> torch.Tensor:
        if self.cos_cached is not None and x.shape[1] <= self.cos_cached.shape[1]:
            return 
        
        seq_len = x.shape[1] 
        theta = 1.0 / self.base ** (torch.arange(0, self.d_model, 2).float() / self.d_model).unsqueeze(0).to(device) # (1, d_model // 2)
        seq_idx = torch.arange(seq_len).unsqueeze(1).to(device) # (seq_len, 1)
        # (seq_len, 1) * (d_model // 2) -> (seq_len, d_model // 2)
        idx_theta = seq_idx * theta # (seq_len, d_model // 2)
        
        # (seq_len, d_model // 2) -> (seq_len, d_model // 2) -> (seq_len, d_model) : concatinate
        idx_theta = torch.cat((idx_theta, idx_theta), dim=-1) # (seq_len, d_model) 
        
        # (seq_len, d_model) -> (seq_len, d_model) -> (seq_len, d_model) : sin and cos
        self.cos_cached = torch.cos(idx_theta[None, :, :]) # (1, seq_len, d_model) : 1 for batch : for broadcasting
        self.sin_cached = torch.sin(idx_theta[None, :, :]) # (1, seq_len, d_model) : 1 for batch : for broadcasting
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._build_cached(x)
        x_cos, x_sin = x[..., 0::2], x[..., 1::2] # (batch, seq_len, d_model) -> (batch, seq_len, d_model // 2)
        # apply rotary positional embedding
        cos_cached, sin_cached = self.cos_cached[:, :, : x_cos.size(-1)], self.sin_cached[:, :, : x_sin.size(-1)] # (1, seq_len, d_model) -> (1, seq_len, d_model // 2)
        
        # (batch, seq_len, d_model // 2) * (1, seq_len, d_model) -> (batch, seq_len, d_model // 2) : broadcasting
        x_cos = x_cos * cos_cached - x_sin * sin_cached
        x_sin = x_cos * sin_cached + x_sin * cos_cached
        x_rot = torch.cat((x_cos, x_sin), dim=-1)   # (batch, seq_len, d_model)
        return x_rot
        

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
        
        self.rotary_emb = RotaryPositionalEmbeddings(d_model)
        
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
        
        # apply rotary positional embedding
        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)
        
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
    
class Roformer(nn.Module): 
    def __init__(self, vocab_len, d_model, n_layers, n_heads, d_ff, dropout, max_len = 1024) -> None: 
        super().__init__() 
        self.embedding = InputEmbedding(vocab_len, d_model)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embedding(x)
        for layer in self.layers: 
            x = layer(x)
        x = self.fc(x)
        return x
    
def init_model(
    model_name = "Roformer",
    d_model = 512,
    n_layers = 3,
    n_heads = 4,
    dropout = 0.15,
    vocab_len = 256
):
    # hyperparameters
    d_ff = 4*d_model     # dimension of feedforward network | 4 times d_model
    roformer = Roformer(vocab_len, d_model, n_layers, n_heads, d_ff, dropout).to(device)
    return roformer


if __name__ == "__main__":
    model = init_model()
    print(model)
    x = torch.randint(0, 256, (2, 512)).to(device)
    y = model(x)
    print(x.shape, y.shape)