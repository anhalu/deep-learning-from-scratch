from .base import Tokenizer, get_pairs, merge_tokens


class BytePairEncoding(Tokenizer): 
    def __init__(self, num_merges: int = 10):
        super().__init__()
        self.num_merges = num_merges
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
            pairs = get_pairs(tokens=tokens) # return counters.
            # get the most frequency pair
            pair, _ = pairs.most_common()[0]
            if _ < 10: # if pair only appear 1 time -> break.
                print("Pair only appear 5 time then break.")
                break
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
    
    