import sys
from pathlib import Path    
print(sys.path)
from llm.tokenizer.bpe import BytePairEncoding 


tokenizer = BytePairEncoding()
tokenizer.load_tokenizer('/home/hoang.minh.an/anhalu-data/learning/deep-learning-from-scratch/llm/tokenizer/tokenizer_vocab_size_10000.json') 


for k, v in tokenizer.vocab.items():
    if k < 128:
        continue
    try:
        v = v.decode('utf-8')
        print(f"Token ID: {k}, Token: {v}")
    except UnicodeDecodeError:
        # print(f"Token ID: {k}, Token: [Unable to decode]")
        pass 