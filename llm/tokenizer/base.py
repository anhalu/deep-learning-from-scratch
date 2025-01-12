from collections import Counter 
from abc import ABC, abstractmethod


class Tokenizer(ABC): 
    @abstractmethod 
    def train(self, text: str, *args, **kwargs): 
        pass
    
    @abstractmethod 
    def encode(self, text: str, *args, **kwargs): 
        pass
    
    @abstractmethod
    def decode(self, tokens: list, *args, **kwargs): 
        pass   


def get_pairs(tokens: list): 
    # get pairs bytes from tokens
    pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)] 
    pairs = Counter(pairs)
    return pairs

def merge_tokens(tokens: list, pair: tuple, new_token): 
    # replace pair with new_id in tokens.
    new_tokens = [] 
    idx = 0 
    while idx < len(tokens): 
        if idx < len(tokens) - 1 and (tokens[idx], tokens[idx + 1]) == pair: 
            new_tokens.append(new_token)
            idx += 2
        else: 
            new_tokens.append(tokens[idx])
            idx +=1
    return new_tokens


if __name__ == "__main__": 
    a = merge_tokens([1, 2, 1,2, 3, 4, 5], (1,2), 10)
    print(a)
