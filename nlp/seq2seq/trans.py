from deep_translator import GoogleTranslator 
import glob 

for file in glob.glob('./data/*.txt'): 
    with open(file, 'r') as f: 
        text = f.read() 
        list_of_text = text.split('\n\n')
        new_pairs = []
        for pairs in list_of_text: 
            pairs = pairs.split('\n') 
            en = pairs[0]
            vi = GoogleTranslator(source='en', target='vi').translate(en) 
            new_pairs.append(f'{en}\t{vi}') 
        
        with open(f'{file}_translated.txt', 'w', encoding='utf-8') as f: 
            f.write('\n'.join(new_pairs)) 


