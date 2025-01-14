from tokenizer import RegexBytePairEncoding 
from data import DataProcessor
import regex as re

tokenizer = RegexBytePairEncoding(vocab_size=10000)

path = 'llm/data/truyen_kieu_data.txt' 
with open(path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer.train(data)
# tokenizer.save_tokenizer(f"tokenizer_regex_vocab_size_{tokenizer.vocab_size}.json") 
# regex_bpe.load_tokenizer('tokenizer_regex.json')
# print("Tokenizer saved to tokenizer.json")
# print(regex_bpe.vocab_size)
tokenizer.load_tokenizer(f"tokenizer_regex_vocab_size_{tokenizer.vocab_size}.json")


for vocab in tokenizer.vocab: 
    print(vocab, tokenizer.decode([vocab]))




# data = """
# 1..Trăm năm trong cõi người ta,
# 2..Chữ tài chữ mệnh khéo là ghét nhau.
# 3..Trải qua một cuộc bể dâu,
# 4..Những điều trông thấy mà đau đớn lòng.
# """


# GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# complied_pattern = re.compile(GPT4_SPLIT_PATTERN)
# split = re.findall(complied_pattern, data)
# print(split)