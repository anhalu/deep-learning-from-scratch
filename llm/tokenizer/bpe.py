from .base import Tokenizer, get_pairs, merge_tokens
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BytePairEncoding(Tokenizer): 
    def __init__(self, vocab_size: int = 255, *args, **kwargs):
        super().__init__()
        assert vocab_size > 255, "vocab_size must be greater than 255."
        self.num_merges = vocab_size - 255 # number of merges.
        self.merges = {} # use for decode.
        self.vocab = {i : bytes([i]) for i in range(256)}  # use for decode.
        self.vocab_size = 255
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
            self.vocab_size += 1
            
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
        # save merges, vocab to file.
        with open(path, 'w') as f:
            for (a, b), c in self.merges.items():
                f.write(f"{a} {b} {c}\n")
            for idx, byte in self.vocab.items():
                f.write(f"{idx} {byte}\n")
    
    def load_tokenizer(self, path: str, *args, **kwargs): 
        # load merges, vocab from file.
        with open(path, 'r') as f:
            for _line in f:
                line = _line.strip().split()
                if len(line) == 3 and 'b' not in _line:
                    a, b, c = map(int, line)
                    self.merges[a, b] = c
                else: 
                    # line = 
                    idx, byte = line[0], ' '.join(line[1:])
                    self.vocab[int(idx)] = bytes(byte, 'utf-8')
                    self.vocab_size = max(self.vocab_size, int(idx))
        self.vocab_size = 255 + self.vocab_size
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
        self.vocab_size = 255
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
        for i in range(self.num_merges): 
            pairs = {}
            for pattern in list_pattern: 
                # get pairs in each pattern
                pairs = get_pairs(pattern, pairs)
            
            # get the most frequency pair
            pair = max(pairs, key=pairs.get)
            new_token = self.vocab_size + i + 1  # i start = 0 then we need + 1.
            self.vocab_size += 1
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
        
        
    
    

if __name__ == "__main__": 
    bpe = BytePairEncoding()
    text = """6..Trời xanh quen thói má hồng đánh ghen.
7..Cảo thơm lần giở trước đèn,
8..Phong tình có lục còn truyền sử xanh.
9,,Rằng năm Gia Tĩnh triều Minh,
10.. Bốn phương phẳng lặng, hai kinh vững vàng.
11..Có nhà viên ngoại họ Vương,
12..Gia tư nghĩ cũng thường thường bực trung."""
    bpe.train(text)
    
    en = bpe.encode('Trời xanh quen thói')
    print(en) 
    de = bpe.decode(en) 
    print(de)
    
    