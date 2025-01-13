import regex as re


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

compiled = re.compile(GPT4_SPLIT_PATTERN)

a = re.findall(compiled, "Xin chào Việt Nam") 
ids = [list(ch.encode('utf-8')) for ch in a]
print(ids)