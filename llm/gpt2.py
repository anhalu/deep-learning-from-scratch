import torch
import math
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


@dataclass
class GPTConfig:
    max_len: int = 1024  # max sequence length
    # number of tokens : 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    vocab_size: int = 50257
    n_layers: int = 12  # number of layers
    n_heads: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension
    dropout: float = 0.0
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
        self.dropout = config.dropout
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
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
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
        return self.weight * (x - mean) / (std + 1e-6) + self.bias if self.bias is not None else self.weight * (x - mean) / (std + 1e-6)


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.max_len, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embed, config.bias)
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
    
        
    @torch.no_grad() 
    def generate(self, tokens, max_new_tokens = 1024, temperature = 1.0, top_k = None): 
        """
        tokens : torch.Tensor, shape (batch, seq_len)
        generate with single text input
        temperature: float, temperature for "soft" sampling from the logits. (softmax with temperature)
        top_k: int, if not None, only the top k tokens will be considered for sampling.
        """
        self.eval()
        tokens_input = tokens if tokens.size(1) <= self.config.max_len else tokens[:, :self.config.max_len]
        for _ in tqdm(range(max_new_tokens)): 
            # forward to 
            logits = self.forward(tokens_input)
            # "soft" sampling from the logits
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None: 
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # (batch, top_k)
                logits = logits.masked_fill(logits < v[:, -1].unsqueeze(-1), float("-inf")) #
            
            # apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            # sample from the distribution
            
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens_input = torch.cat([tokens_input, idx_next], dim=-1)
        
        return tokens_input
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Load parameters from pre-trained model.
        """
        
        assert model_type in ['gpt2'], "Model type must be 'gpt2'"
        
        config = GPTConfig() 
        model = GPT2(config) 
        
        sd = model.state_dict()
        
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        hf_sd_keys = list(hf_sd.keys())
        
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        for k, v in sd.items(): 
            if k in hf_sd_keys: 
                # if the key is in the hf model, we can just copy it over
                # if the key in transposed, we need to transpose the weight.
                if any(k.endswith(w) for w in transposed):
                    # check shape. 
                    assert v.shape == hf_sd[k].T.shape, f"Shape mismatch key transposed {k}: {v.shape} and {hf_sd[k].T.shape}"
                    with torch.no_grad():
                        sd[k].copy_(hf_sd[k].T)
                else:
                    assert v.shape == hf_sd[k].shape, f"Shape mismatch key {k}: {v.shape} and {hf_sd[k].shape}"
                    with torch.no_grad():
                        sd[k].copy_(hf_sd[k])
            else: 
                print(f"Key {k} not found in hf model")
        
        model.load_state_dict(sd)
        print("Model loaded successfully")
        return model
    

    # @classmethod
    # def from_pretrained(cls, model_type, override_args=None):
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     override_args = override_args or {} # default to empty dict
    #     # only dropout can be overridden see more notes below
    #     assert all(k == 'dropout' for k in override_args)
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':         dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
    #         'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 350M params
    #         'gpt2-large':   dict(n_layers=36, n_heads=20, n_embed=1280), # 774M params
    #         'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embed=1600), # 1558M params
    #     }[model_type]
    #     print("forcing vocab_size=50257, block_size=1024, bias=True")
    #     config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    #     config_args['max_len'] = 1024 # always 1024 for GPT model checkpoints
    #     config_args['bias'] = True # always True for GPT model checkpoints
    #     # we can override the dropout rate, if desired
    #     if 'dropout' in override_args:
    #         print(f"overriding dropout rate to {override_args['dropout']}")
    #         config_args['dropout'] = override_args['dropout']
    #     # create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT2(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     # for k in sd_keys:
    #     #     assert k in sd_keys_hf, f"missing key {k}"
    #     # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}" 
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

if __name__ == '__main__':
    config = GPTConfig()
    model = GPT2(config).from_pretrained("gpt2")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    list_text = [
        "Hi, My name is ",
        "I am a transformer model",
        "I am a GPT-2 model"
    ]
    # encode the text to tokens and add padding to get batch of tokens
    tokenizer.pad_token = tokenizer.unk_token
    tokens = tokenizer(list_text, padding=True, return_tensors="pt")
    
    # generate new tokens
    new_tokens = model.generate(tokens["input_ids"], max_new_tokens=100)
    # print(new_tokens)
    for seq in new_tokens: 
        print(tokenizer.decode(seq))
    
