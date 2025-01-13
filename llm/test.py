from tokenizer import RegexBytePairEncoding 
from data import DataProcessor


# regex_bpe = RegexBytePairEncoding(vocab_size=1000)

# path = 'llm/data/truyen_kieu_data.txt' 
# with open(path, 'r', encoding='utf-8') as f:
#     data = f.read()

# # regex_bpe.train(data)
# # regex_bpe.save_tokenizer('tokenizer_regex.json') 
# regex_bpe.load_tokenizer('tokenizer_regex.json')
# # print("Tokenizer saved to tokenizer.json")
# print(regex_bpe.vocab_size)



tokenizer = RegexBytePairEncoding()
tokenizer.load_tokenizer("tokenizer_regex.json")

# load data and create dataloader
data_processor = DataProcessor(tokenizer=tokenizer) 
train_loader, test_loader = data_processor.get_dataloaders()
print(len(train_loader.dataset))