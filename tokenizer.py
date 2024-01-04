import collections
from tqdm import tqdm 
import datasets 
import string 
import os 
import re
import json
import torch

class BPE():
    def __init__(self, path_list, training_bool : bool = False,  training_steps = 1000, openwebtext_10k_bool : bool = False, longform_bool : bool = False):

        #SPECIAL TOKENS
        self.begin_token = '<begin>'
        self.speaker_change_token = '<spkchg>'
        #self.end_sentence_token = '</s>'

        self.unknown_token = '<unk>'
        self.end_token = '</w>'
        self.allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:-!?()' ")
        
       

        if training_bool :

            self.path_list = path_list
            self.text_corpus = ''
            print("Loading dialogue dataset ...")
            
            self.load_text_files()
            if openwebtext_10k_bool :
                print("Loading Openwebtext dataset ...")
                dataset = datasets.load_dataset('stas/openwebtext-10k')['train']['text']

                
                self.text_corpus += self.begin_token.join(dataset)               
                #print(''.join(dataset))
            elif longform_bool :
                print("Loading Longform dataset ...")
                
                for split in ['train', 'validation', 'test'] :
                    dataset = datasets.load_dataset("akoksal/LongForm")[split]
                    with tqdm(total = len(dataset), desc = f'loading {split} split') as pbar :
                        for sample in dataset :

                            if sample['source'] != 'StackExchange':
                                line = sample['output'].replace(" ’ ", "'").replace('—','-').replace('#','').replace('、','').replace('¥', '$').replace('£', '$').replace('°','').replace('\\','').replace('“','"').replace('”', '"').replace('‘', "'").replace('~','').replace('’',"'").replace('′', "'").replace('。', '').replace('_','').replace('@','').replace('–','-').replace('\n', ' ')

                                line = self.replace_chars(line, self.allowed_chars, replacement_char='')    
                                self.text_corpus += self.begin_token+line
                            pbar.update(1)

                
            self.vocab = self.create_vocab_from_file()
        
            self.train(training_steps)
            self.saving()
        else :
            self.load()

        #ADDING SPECIAL TOKENS
        self.tokens = self.tokens 
        self.itos = {i: s for i, s in enumerate(self.tokens)}
        self.stoi = {s: i for i, s in enumerate(self.tokens)}

    def replace_chars(self, input_string, allowed_chars, replacement_char):
        """
        Replace characters in input_string that are not in allowed_chars with replacement_char.

        Parameters:
        input_string (str): The string to process.
        allowed_chars (set): A set of characters that are allowed to remain unchanged.
        replacement_char (str): The character to replace disallowed characters with.

        Returns:
        str: The modified string.
        """
        return ''.join([c if c in allowed_chars else replacement_char for c in input_string])



    def load_text_files(self,):

# 
        for path in self.path_list:
            text = []
            if path.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                       
                            line = line.strip()

                            line = line.replace(" ’ ", "'").replace('—','-').replace('、','').replace('°','').replace('\\','').replace('“',"'").replace('”', "'").replace('‘', "'").replace('~','').replace('’',"'").replace('′', "'").replace('–','-')
                            line = line.replace(" .", ".").replace(" !", "!").replace(" ,", ",").replace(" ?", "?").replace(" ;",";")
                            line = self.replace_chars(line, self.allowed_chars, replacement_char='')     

                            text.append(line)
                 
                self.text_corpus += ''.join(text)

    


    def get_pairs(self):
        paris = collections.defaultdict(int)

        for tokens, count in self.vocab.items():
            chars = tokens.split()

            for i in range(len(chars) - 1):
                paris[(chars[i], chars[i+1])] += count

        return paris

                

    def create_vocab_from_file(self,) -> collections.defaultdict:
        vocab = collections.defaultdict(int)
        #print(self.text_corpus)
        words = self.text_corpus.split()
        # words = re.split(r'( )|\n', self.text_corpus)
        # words = [word for word in words if word]
        with tqdm(total=len(words)) as pbar:
            for word in words:
                keeping = True
                for char in word : 
                    if char not in string.printable:
                        keeping = False
                # if '電' in word or '令' in word or '≥' in word :
                #     print(word)
                if keeping :
                    vocab[' '.join(list(word)) + f' {self.end_token}'] += 1

                pbar.update(1)

        return vocab
    

    def merge_tokens_and_update_vocab(self, pair):
        temp_vocab = {}


        pattern = re.compile(
            r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
        for token in self.vocab:
            updated_token = pattern.sub(''.join(pair), token)
            temp_vocab[updated_token] = self.vocab[token]

        self.vocab = temp_vocab

    def sort_tokens(self, tokens_with_freq: dict) -> list:
        if not isinstance(tokens_with_freq, dict):
            raise TypeError("`tokens_with_freq` must be a `dict`")

        # Sort tokens by length
        sorted_tokens = sorted(tokens_with_freq.items(),
                               key=lambda x: (len(x[0]), x[1]), reverse=True)
        sorted_tokens = [token for token, _ in sorted_tokens]

        return sorted_tokens

    def get_tokens_from_vocab(self, vocab: dict) -> list:

        frequencies = collections.defaultdict(int)
        words_to_tokens = {}
        
        for word, count in vocab.items():
            chars = word.split()
            
            for char in chars:
                frequencies[char] += count
            words_to_tokens[''.join(chars)] = chars
        return frequencies, words_to_tokens
    

    def train(self, iterations: int) -> None:
        if not isinstance(iterations, int):
            raise TypeError("iterations must be int")

        with tqdm(total=iterations) as pbar:
            for i in range(iterations):

                token_pairs = self.get_pairs()

                # stop if there are no more pairs
                if not token_pairs:
                    break

                # get the most frequent pair, marge those, create new tokens
                # and update the vocab
                most_frequent_pair = max(token_pairs, key=token_pairs.get)
                self.merge_tokens_and_update_vocab(most_frequent_pair)
                pbar.update(1)

        # finally, get the tokens and frequencies
   
        tokens, words_to_tokens = self.get_tokens_from_vocab(self.vocab)
        self.tokens = self.sort_tokens(tokens)
        self.words_to_tokens = words_to_tokens

        print("Training finished")
            
    def tokenize(self, text, pbar_bool : bool = True) -> list:
        txt = []
        text = text.split()
       
        if pbar_bool :
            with tqdm(total = len(text), desc = 'Tokenizing the text')as pbar : 
                for word in text:
                    txt = txt + self.word_tokenize(word) 
                    pbar.update(1)
        else :
            for word in text:
                    txt = txt + self.word_tokenize(word) 
            
        return txt 
    


    
        
    def word_tokenize(self, word) -> list:
        word += self.end_token
        return self.__tokenize(word)

    def __tokenize(self, word: str, tokens: list = None) -> list:
        if tokens is None:
            tokens = self.tokens
        # if the word is in the vocab, return the word
        # if word in self.words_to_tokens:
        #     return self.words_to_tokens[word]
        if word == "":
            return []
        if tokens == []:
            return [self.unknown_token]

        out_tokens = []
        for i in range(len(self.tokens)):
            
            temp_token = self.tokens[i]
            esc_token = re.escape(temp_token.replace('.', '[.]'))

            # Check if the token is in the word. If so, get the indices of the
            # matched portion
            matched_indices = [(matched.start(0), matched.end(0))
                               for matched in re.finditer(esc_token, word)]
            if len(matched_indices) == 0:
                # This token is not present in the word
                continue

            # The token is present int the word. Get the end position of the
            # matched portion. There can be several matched positions.
            subword_end_indices = [matched_index[0]
                                   for matched_index in matched_indices]
            subword_start_index = 0

            for end_index in subword_end_indices:
                subword = word[subword_start_index:end_index]
                # tokenize next subwords and append them to the output tokens
                out_tokens.extend(self.__tokenize(subword, self.tokens[i+1:]))
                out_tokens.append(temp_token)

                # update the start index for the last subword
                subword_start_index = end_index + len(temp_token)

            # get the remaining subword
            remaining_subword = word[subword_start_index:]
            out_tokens.extend(self.__tokenize(
                remaining_subword, self.tokens[i+1:]))
            break
        return out_tokens
    
    def saving(self,):
        with open('words_to_tokens.json', 'w') as fp:
            json.dump(self.words_to_tokens, fp)
        with open('tokens.json', 'w') as fp:
            json.dump(self.tokens, fp)

    def load(self,):
        with open('words_to_tokens.json', 'r') as fp:
            self.words_to_tokens = json.load(fp)
        with open('tokens.json', 'r') as fp:
            self.tokens = json.load(fp)

        self.itos = {i: s for i, s in enumerate(self.tokens)}
        self.stoi = {s: i for i, s in enumerate(self.tokens)}

    def encode(self, text) :
        encoded_text = [self.stoi[token] for token in text]
        return encoded_text

    def decode(self, encoded_text) :
        if isinstance(encoded_text, torch.Tensor):
            encoded_text = encoded_text.tolist()
        text = [self.itos[token] for token in encoded_text]
        text = ''.join(text)
        text = text.replace('</w>', ' ')
        return text



    # def decode(self, ids):
    #     if isinstance(ids, torch.Tensor):
    #         ids = ids.tolist()
    #     return ''.join([self.itos[id] for id in ids])
    
    # def encode(self, text):
    #     return [self.stoi[char] for char in text if char in self.stoi]

if __name__ == '__main__':
    path_list = []
    for split_str in ['train', 'test', 'validation']:
        for idx, file in enumerate(os.listdir('dataset/dialogue_dataset/'+str(split_str))):
            path_list.append('dataset/dialogue_dataset/'+str(split_str)+'/'+file)
    #vocab = BPE.create_vocab_from_file(path_list)
    tokenizer = BPE(path_list,training_bool = True, openwebtext_10k_bool=False, longform_bool = True)
    #tokenizer.load()
    tokenized_text = tokenizer.tokenize('Bonjour, je suis un test')
    print(tokenized_text)
    encoded_text = tokenizer.encode(tokenized_text)
    print(encoded_text)
    print(tokenizer.decode(encoded_text))
    # encoded_text = tokenizer.encode('Bonjour, je suis un test')
    # print(encoded_text)

