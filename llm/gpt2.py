import torch
import math
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    max_len: int = 1024  # max sequence length
    # number of tokens : 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    vocab_size: int = 50257
    n_layers: int = 12  # number of layers
    n_heads: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension
    dropout: float = 0.1
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = True

class MLP(nn.Module): 
    def __init__(self, config: GPTConfig) -> None: 
        super().__init__() 
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c_fc(x))
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config : GPTConfig) -> None: 
        super().__init__()
        assert config.n_embed % config.n_heads == 0, "n_embed must be divisible by n_heads"

        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.droupout = config.dropout
        
        self.head_dim = config.n_embed // config.n_heads
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias) # matix for queries, keys, values

        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash: 
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, T, C = x.shape # (batch, seq_len, n_embed)
        
        # linear transformation for queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1) # (batch, seq_len, n_embed)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (batch, n_heads, seq_len, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # same
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # same
        
        # causal self-attention : (B, n_heads, seq_len or T, head_dim) * (B, n_heads, head_dim, seq_len or T) -> (B, n_heads, seq_len, seq_len)
        if self.flash: 
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, self.attn_dropout)
        else: 
            # mauallly implement causal attention 
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, n_heads, seq_len, seq_len)
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            attn_score = torch.softmax(attn, dim=-1)
            attn_score = self.attn_dropout(attn_score)
            y = torch.matmul(attn_score, v) # (B, n_heads, seq_len, head_dim)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embed) # (B, seq_len, n_embed)
        
        # linear transformation for the concatenated outputs
        y = self.c_proj(y) # (B, seq_len, n_embed)
        
        return y
            

class Block(nn.Module): 
    def __init__(self, config: GPTConfig) -> None: 
        super().__init__() 
        self.ln_1 = LayerNorm(config.n_embed, config.bias) 
        self.attn = CausalMultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LayerNorm(nn.Module): 
    def __init__(self, d_model: int, bias = None) -> None: 
        super().__init__() 
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + 1e-6) + self.bias


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.max_len, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embed)
        ))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.config.max_len, "Sequence length is too long"
        
        # word token embedding + positional embedding
        h = self.transformer['wte'](x) + self.transformer['wpe'](torch.arange(T, device=x.device))
        
        # transformer blocks
        for block in self.transformer['h']:
            h = block(h)
        
        # final layer norm
        h = self.transformer['ln_f'](h)
        
        # language model head
        lm_logits = self.lm_head(h)
        
        return lm_logits
        


if __name__ == '__main__':
    config = GPTConfig()
    model = GPT2(config)
    sd = model.state_dict() 
    for k, v in sd.items():
        print(k, v.shape)
