from .base import Tokenizer, get_pairs, merge_tokens
import regex as re
import json
from tqdm import tqdm
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BytePairEncoding(Tokenizer): 
    def __init__(self, vocab_size: int = 300, *args, **kwargs):
        super().__init__()
        assert vocab_size > 255, "vocab_size must be greater than 255."
        self.num_merges = vocab_size - 255 # number of merges.
        self.merges = {} # use for decode.
        self.vocab = {i : bytes([i]) for i in range(256)}  # use for decode.
        self.vocab_size = vocab_size
        # an byte range (0, 255 - 8 bit)
        
    def train(self, text, *args, **kwargs):
        """
        Byte level use utf-8
            merges : (int a, int b) -> int c. merge a, b to c.
            vocab : [idx] -> byte
        """
        text = text.encode('utf-8') # raw text.
        tokens = list(text) # list of each byte (int) in utf-8
        
        for i in range(self.num_merges): 
            pairs = get_pairs(tokens=tokens) # return pairs (a, b) -> freq.
            # get the most frequency pair
            pair = max(pairs, key = pairs.get)
            
            # update merges, vocab -> update tokens. 
            new_token = 256 + i # one byte range (0, 255)
            # self.vocab_size += 1
            
            self.merges[pair[0], pair[1]] = new_token
            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]] # byte1 + byte2. for decode.
            
            # replace pair in tokens by new token. 
            tokens = merge_tokens(tokens, pair, new_token)
        
    def encode(self, text: str, *args, **kwargs) -> list:
        # get list of idx in utf-8.
        tokens = list(text.encode('utf-8'))
        while len(tokens) > 2:  # tokens <= 2 -> one character or empty
            # get pairs in tokens. 
            pairs = get_pairs(tokens)
            pairs = [pair for pair, _ in pairs] # ignore counter.
            
            # get the lowest key in merges then merge tokens, continue utils can not merge.
            pair = min(pairs, key = lambda p: self.merges.get(p, float("inf")))
            # case pair not in merges -> break.
            if pair not in self.merges:
                break
            new_token = self.merges[pair]
            # merge and update tokens. 
            tokens = merge_tokens(tokens, pair, new_token)
        
        return tokens
            
    
    def decode(self, tokens: list, *args, **kwargs) -> str:
        # with each idx in tokens then get bytes from idx in vocab then decode with utf-8 
        byte_raw = b''.join(self.vocab[idx] for idx in tokens) 
        text = byte_raw.decode('utf-8', errors='replace') 
        return text
    
    def save_tokenizer(self, path: str, *args, **kwargs):
        # save with json format. 
        merges_str_keys = {str(k): v for k, v in self.merges.items()}
        vocab_serializable = {k: list(v) for k, v in self.vocab.items()}
        
        with open(path , 'w', encoding='utf-8') as f:
            json.dump({
                "merges": merges_str_keys,
                "vocab": vocab_serializable
            }, f)
        
    def load_tokenizer(self, path: str, *args, **kwargs):
        # load with json format. 
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys back to tuples and lists of integers back to bytes
            self.merges = {eval(k): v for k, v in data['merges'].items()}
            self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
            self.vocab_size = max(self.vocab.keys())
        return self
                  
                  

class RegexBytePairEncoding(Tokenizer):
    """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
    """
    def __init__(self, vocab_size = 512, *args, **kwargs):
        super().__init__()
        assert vocab_size > 255, "vocab_size must be greater than 255."
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 255
        self.merges = {} # use (a, b) -> c for decode.
        self.vocab = {i: bytes([i]) for i in range(255)}  # use idx -> byte for decode.
        # an byte range (0, 255 - 8 bit)
        
        self.special_tokens = {} 
        self.reverse_special_tokens = {}
        self.pattern = kwargs.get('pattern', GPT4_SPLIT_PATTERN)
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text: str, *args, **kwargs):
        """
        Use regex to split text into tokens first than train bpe like normal.
        """
        
        list_pattern = re.findall(self.compiled_pattern, text)
        list_pattern = [list(pattern.encode('utf-8')) for pattern in list_pattern]
        for i in tqdm(range(self.num_merges), desc="Merges"): 
            pairs = {}
            for pattern in list_pattern: 
                # get pairs in each pattern
                pairs = get_pairs(pattern, pairs)
            
            # get the most frequency pair
            pair = max(pairs, key=pairs.get)
            new_token = 256 + i   # i start = 0 then we need + 1.

            self.merges[pair[0], pair[1]] = new_token
            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            list_pattern = [
                merge_tokens(pattern, pair, new_token) for pattern in list_pattern
            ]
    
    
    def register_special_tokens(self, special_tokens: dict):
        """
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.reverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def origin_encode(self, text: str, *args, **kwargs) -> list:
        # get list of idx in utf-8.
        tokens = list(text.encode('utf-8'))
        while len(tokens) > 2:  # tokens <= 2 -> one character or empty
            # get pairs in tokens. 
            pairs = get_pairs(tokens)
            pairs = [pair for pair, _ in pairs] # ignore counter.
            
            # get the lowest key in merges then merge tokens, continue utils can not merge.
            pair = min(pairs, key = lambda p: self.merges.get(p, float("inf")))
            # case pair not in merges -> break.
            if pair not in self.merges:
                break
            new_token = self.merges[pair]
            # merge and update tokens. 
            tokens = merge_tokens(tokens, pair, new_token)
        
        return tokens
    
    def regex_encode(self, text: str, *args, **kwargs) -> list: 
        list_pattern = re.findall(self.compiled_pattern, text) 
        tokens = []
        for pattern in list_pattern: 
            tokens.extend(self.origin_encode(pattern))
        return tokens
    
    def encode(self, text:str, allowed_special="none_raise"): 
        """
        Handle special tokens. 
        allowed_speical: Can be "all|none|non_raise" or custom set of speical tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun

        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special: 
            return self.origin_encode(text) 
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        
        tokens = []
        for chunk in special_chunks:
            if chunk in special:
                tokens.append(special[chunk])
            else:
                tokens.extend(self.origin_encode(chunk))
        
        return tokens
        
    
    def decode(self, tokens: list, *args, **kwargs) -> str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in tokens:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.reverse_special_tokens:
                part_bytes.append(self.reverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save_tokenizer(self, path: str, *args, **kwargs):
        # save with json format. 
        merges_str_keys = {str(k): v for k, v in self.merges.items()}
        vocab_serializable = {k: list(v) if isinstance(v, bytes) else v for k, v in self.vocab.items()}
        special_tokens_serializable = {k: list(v) if isinstance(v, bytes) else v for k, v in self.special_tokens.items()}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "merges": merges_str_keys,
                "vocab": vocab_serializable,
                "special_tokens": special_tokens_serializable
            }, f)
            
    def load_tokenizer(self, path: str, *args, **kwargs):
        # load with json format. 
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys back to tuples and lists of integers back to bytes
            self.merges = {eval(k): v for k, v in data['merges'].items()}
            self.vocab = {int(k): bytes(v) if isinstance(v, list) else v for k, v in data['vocab'].items()}
            self.special_tokens = {k: v for k, v in data['special_tokens'].items()}
            self.reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
            self.vocab_size = len(self.vocab) + len(self.special_tokens) 
        return self