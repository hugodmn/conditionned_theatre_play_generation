import os 
import pandas as pd 
import re 
from model_V2 import Config
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
from tokenizer import BPE 
import pickle
import ast

allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:-!?()' <>")

def replace_chars( input_string, allowed_chars, replacement_char):
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

class Longform_BPE(Dataset):
    def __init__(self,ctx_len : int = 256, split_str : str = "train", dataset_alreday_tokenized = True):


        self.tokenizer = BPE('dataset/dialogue_dataset', training_bool = False)
        self.itos = self.tokenizer.itos
        self.stoi = self.tokenizer.stoi
        self.vocab_size = len(self.stoi)

        self.ctx_len = ctx_len

        self.tokenized_text = []
        if not dataset_alreday_tokenized : 

            for split in ['train', 'validation', 'test'] :
                text = ""
                dataset = load_dataset("akoksal/LongForm")[split]

                if split == 'train':
                    dataset = dataset[:10000]
                if split == 'test' : 
                    dataset = dataset[:500]
                if split == 'validation':
                    dataset = dataset[:1000]

                with tqdm(total = len(dataset['input']), desc = f'Loading Longform {split} split') as pbar :
                        for sample_idx in range(len(dataset['input'])) :
                            line = dataset['output'][sample_idx].replace(" ’ ", "'").replace('—','-').replace('#','').replace('、','').replace('¥', '$').replace('£', '$').replace('°','').replace('\\','').replace('“','"').replace('”', '"').replace('‘', "'").replace('~','').replace('’',"'").replace('′', "'").replace('。', '').replace('_','').replace('@','').replace('–','-').replace('\n', ' ')
                            line = replace_chars(input_string=line, allowed_chars=allowed_chars, replacement_char='')
                            text += self.tokenizer.begin_token + line
                            pbar.update(1)
                text = self.tokenizer.tokenize(text, pbar_bool=True)
                with open(f"dataset/longform_{split}", "wb") as f:
                    pickle.dump(text, f)
          
        else  :
            with open(f"dataset/longform_{split_str}", "rb") as f:
                self.tokenized_text = pickle.load(f)
            
            print('Dataset loaded.')
      
        # if split_str == 'train' :
        #     self.tokenized_text = self.tokenized_text[:int(self.__len__()*rate_to_keep)]
        # elif split_str == "test":
        #     self.tokenized_text = self.tokenized_text[int(self.__len__()*rate_to_keep):]
    
    def __len__(self):
        return len(self.tokenized_text) - self.ctx_len - 1

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to integer
        input_seq = self.encode(self.tokenized_text[idx:idx + self.ctx_len])
        target_seq = self.encode(self.tokenized_text[idx + 1:idx + 1 + self.ctx_len])

        return torch.tensor(input_seq), torch.tensor(target_seq)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)

class OpenWebText_BPE(Dataset):
    def __init__(self,ctx_len : int = 256, split_str : str = "train", rate_to_keep : float = 0.9, dataset_alreday_tokenized = True):


        self.tokenizer = BPE('dataset/dialogue_dataset', training_bool = False)
        self.itos = self.tokenizer.itos
        self.stoi = self.tokenizer.stoi
        self.vocab_size = len(self.stoi)

        self.processor = OWTProcessData(ctx_len)
        self.ctx_len = ctx_len

        self.tokenized_text = []
        if not dataset_alreday_tokenized : 
            not_processed_dataset = load_dataset("stas/openwebtext-10k")['train']['text']

            with tqdm(total=len(not_processed_dataset), desc = "Tokenizing the openwebtext dataset") as pbar : 
                with open('dataset/openwebtext_tokenized.txt', 'w') as f:
                    for text in not_processed_dataset : 
                        self.tokenized_text = self.tokenizer.tokenize(self.tokenizer.begin_token+text)
                        f.write(str(self.tokenized_text)+'\n')
                        pbar.update(1)
        else  :
            print('Loading dataset...')
            with open('dataset/openwebtext_tokenized.txt', 'r') as openwebtext_tokenized:
                lines = openwebtext_tokenized.readlines()
                if split_str == 'train' :
                  with tqdm(total=len(lines[:int(len(lines)*rate_to_keep)]), desc = f"Loading {split_str} split") as pbar :
                    for line in lines[:int(len(lines)*rate_to_keep)] : 
                    #line = json.loads(line)
                        line = ast.literal_eval(line)
                
                        self.tokenized_text = self.tokenized_text + line
                        pbar.update(1)

                elif split_str == "test":
                    with tqdm(total=len(lines[int(len(lines)*rate_to_keep):]), desc = f"Loading {split_str} split") as pbar :
                        for line in lines[int(len(lines)*rate_to_keep):] : 
                        #line = json.loads(line)
                            line = ast.literal_eval(line)
                    
                            self.tokenized_text = self.tokenized_text + line
                            pbar.update(1)
            print('Dataset loaded.')
      
        # if split_str == 'train' :
        #     self.tokenized_text = self.tokenized_text[:int(self.__len__()*rate_to_keep)]
        # elif split_str == "test":
        #     self.tokenized_text = self.tokenized_text[int(self.__len__()*rate_to_keep):]
    
    def __len__(self):
        return len(self.tokenized_text) - self.ctx_len - 1

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to integer
        input_seq = self.encode(self.tokenized_text[idx:idx + self.ctx_len])
        target_seq = self.encode(self.tokenized_text[idx + 1:idx + 1 + self.ctx_len])

        return torch.tensor(input_seq), torch.tensor(target_seq)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)

class DialogueDataset_BPE(Dataset):
    def __init__(self, split_str : str = 'train', ctx_len : int = 256, already_tokenized = True):
        self.tokenizer = BPE('dataset/dialogue_dataset', training_bool = False)
        self.itos = self.tokenizer.itos
        self.stoi = self.tokenizer.stoi 
        self.vocab_size = len(self.stoi)
        if not already_tokenized :
            train_corpus = ""
            test_corpus = ""
            validation_corpus = ""
            for split in ['train', 'test', 'validation']:

                with tqdm(total = len(os.listdir('dataset/dialogue_dataset/'+split)), desc = f'loading {split} split') as pbar : 
                    for idx, file in enumerate(os.listdir('dataset/dialogue_dataset/'+split)):
                    
                        if file.endswith('.txt'):
                            with open('dataset/dialogue_dataset/'+split+'/'+file, 'r') as f:
                                lines = f.readlines()
                                line = lines[0]
                                line = line.strip()
                                line = line.replace(" ’ ", "'").replace('—','-').replace('#','').replace('、','').replace('¥', '$').replace('£', '$').replace('°','').replace('\\','').replace('“','"').replace('”', '"').replace('‘', "'").replace('~','').replace('’',"'").replace('′', "'").replace('。', '').replace('_','').replace('@','').replace('–','-').replace('\n', ' ')
                                line = line.replace('<spkchg>', ' <spkchg> ')
                                line = replace_chars(input_string=line, allowed_chars=allowed_chars, replacement_char='')

                                if split == 'train':
                                    train_corpus = train_corpus + line
                                elif split == 'test':
                                    test_corpus = test_corpus + line
                                elif split == 'validation':
                                    validation_corpus = validation_corpus + line
                        pbar.update(1)
            
            tokenized_trainset = self.tokenizer.tokenize(train_corpus)
            tokenized_validset = self.tokenizer.tokenize(validation_corpus)
            tokenized_testset = self.tokenizer.tokenize(test_corpus)

            for split in ['tokenized_trainset', 'tokenized_validset', 'tokenized_testset']:

                
                    if split == 'tokenized_trainset':
                        with open(f'dataset/{split}_dialogue_dataset', 'wb') as f:
                            pickle.dump(tokenized_trainset, f)
                    elif split == 'tokenized_validset':
                        with open(f'dataset/{split}_dialogue_dataset', 'wb') as f:
                            pickle.dump(tokenized_validset, f)
                    elif split == 'tokenized_testset':
                        with open(f'dataset/{split}_dialogue_dataset', 'wb') as f:
                            pickle.dump(tokenized_testset, f)
          
            print("TOKENIZED DATASET IS SAVED !")

        else : 

            with open(f'dataset/{split_str}_dialogue_dataset', 'rb') as f:
                self.dataset = pickle.load(f)



        self.ctx_len = ctx_len

    def __len__(self):
        return len(self.dataset) - self.ctx_len - 1
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to integer
        input_seq = self.encode(self.dataset[idx:idx + self.ctx_len])
        target_seq = self.encode(self.dataset[idx + 1:idx + 1 + self.ctx_len])

        return torch.tensor(input_seq), torch.tensor(target_seq)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    

    def process_dataset(self, saving = True):

        self.train_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        self.val_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        self.test_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        for split in ['train', 'test' , 'validation']:
            
            
                with open('dataset/'+self.dataset_path+'/'+split+'/dialogues_'+split+'.txt', 'r') as dialogues_f:
                    # if split == 'train':
                    #     dialogues_lines = dialogues_f.readlines()[:4000]
                    # if split == 'test':
                    #     dialogues_lines = dialogues_f.readlines()[
                    dialogues_lines = dialogues_f.readlines()


                    with tqdm(total=len(dialogues_lines), desc=f"Creating samples for {split} split") as pbar:
                        for idx in range(len(dialogues_lines)) :
                            dialogue = ""
                            dialogue_line = dialogues_lines[idx].split('__eou__')[:-1]
                            #repartition_line = [int(char) for char in repartition_lines[idx] if char != '\n' and char != ' ']
                            dialogue += '<begin>'
                     
                            for sentence_idx in range(len(dialogue_line)) :
                                if sentence_idx != 0 :
                                    dialogue += '<spk_chg>'
                                dialogue += dialogue_line[sentence_idx][:-1].strip()
                            if len(self.tokenizer.tokenize(dialogue)) > self.ctx_len+50 :
                                with open('dataset/dialogue_dataset/'+split+'/dialogues_'+split+'_'+str(idx)+'.txt', 'w') as f:
                                    f.write(dialogue)

                            pbar.update(1)   


        







class DialogueDataset_char(Dataset):
    def __init__(self, dataset_path : str, split_str : str, ctx_len : int = 256, saving = False):
        self.tokenizer = BPE('dataset/dialogue_dataset', training_bool = False)
        self.dataset_path = dataset_path
        self.ctx_len = ctx_len
        self.vocab_chars = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', '[', '\n', ']', '{', '}', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        self.vocab_size = len(sorted(self.vocab_chars))
        self.stoi = {char: i for i, char in enumerate(sorted(self.vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(self.vocab_chars))}
        #self.create_dataset(saving = True)
        #self.generate_csv_sample_mapping()
        compt = 0
        self.dataset = pd.DataFrame(columns = ['id','file', 'dialogue'])

    def __len__(self):
        return len(self.dataset)
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.itos[id] for id in ids])
    
    def encode(self, text):
        return [self.stoi[char] for char in text]
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        dialogue = self.dataset.at[idx, 'dialogue']

        start_index = random.randint(0, len(dialogue) - self.ctx_len - 1)
        input_seq = dialogue[start_index:start_index + self.ctx_len]
        target_seq = dialogue[start_index + 1:start_index + 1 + self.ctx_len]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)
    def create_dataset(self, saving = True):

        self.train_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        self.val_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        self.test_df = pd.DataFrame(columns = ['path', 'sample_nb'])
        for split in ['train', 'test' , 'validation']:
            
            
                with open('dataset/'+self.dataset_path+'/'+split+'/dialogues_'+split+'.txt', 'r') as dialogues_f:
                    # if split == 'train':
                    #     dialogues_lines = dialogues_f.readlines()[:4000]
                    # if split == 'test':
                    #     dialogues_lines = dialogues_f.readlines()[
                    dialogues_lines = dialogues_f.readlines()


                    with tqdm(total=len(dialogues_lines), desc=f"Creating samples for {split} split") as pbar:
                        for idx in range(len(dialogues_lines)) :
                            dialogue = ""
                            dialogue_line = dialogues_lines[idx].split('__eou__')[:-1]
                            #repartition_line = [int(char) for char in repartition_lines[idx] if char != '\n' and char != ' ']
                            dialogue += '<begin>'
                     
                            for sentence_idx in range(len(dialogue_line)) :
                                if sentence_idx != 0 :
                                    dialogue += '<spk_chg>'
                                dialogue += dialogue_line[sentence_idx][:-1].strip()
                            if len(self.tokenizer.tokenize(dialogue)) > self.ctx_len+50 :
                                with open('dataset/dialogue_dataset/'+split+'/dialogues_'+split+'_'+str(idx)+'.txt', 'w') as f:
                                    f.write(dialogue)

                            pbar.update(1)   



    def generate_csv_sample_mapping(self,):
        mapping_df = pd.DataFrame(columns = ['idx', 'dialogue'])
        for split in ['train', 'test' , 'validation']:
            compt = 0
            with tqdm(total=len(os.listdir('dialogue_dataset/'+split)), desc=f"Creating csv for {split} split") as pbar:
                for idx, file in enumerate(os.listdir('dialogue_dataset/'+split)):
                    if file.endswith('.txt'):
                        with open('dialogue_dataset/'+split+'/'+file, 'r') as f:
                            lines = f.readlines()
                            corpus = ''.join(lines)
                            for idx in range(len(corpus)-self.ctx_len-1) :
                                mapping_df.at[compt, 'idx'] = idx
                                mapping_df.at[compt, 'dialogue'] = file

                                compt += 1
                    pbar.update(1)
            mapping_df.to_csv('dialogue_dataset/'+split+'/repartition.csv')
        








class TheatreDataset(Dataset):

    def __init__(self, split_str : str, full_txt_path, emo : bool = False, instruct = False, ctx_len = 512, saving = False):

        self.vocab_chars = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', '[', '\n', ']', '{', '}', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        self.vocab_size = len(sorted(self.vocab_chars))
        self.stoi = {char: i for i, char in enumerate(sorted(self.vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(self.vocab_chars))}

        self.ctx_len = ctx_len

        if saving :
            self.load_txt_datasets(full_txt_path, 'characters.csv', emo, instruct, saving = saving)

       
        
    
        if split_str == 'train':
            self.df_dataset = pd.read_csv('train_df.csv')
            
        elif split_str == 'val':
            self.df_dataset = pd.read_csv('val_df.csv')
            
        
        
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to integer
        
        inputs = self.df_dataset.iloc[idx]['input']
        targets = self.df_dataset.iloc[idx]['target']
        inputs = self.encode(inputs)
        targets =  self.encode(targets)
        return torch.tensor(inputs), torch.tensor(targets)
    
    def __len__(self):
        return len(self.df_dataset) 


    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.itos[id] for id in ids])
    
    def encode(self, text):
        return [self.stoi[char] for char in text if char in self.stoi]
    
    def load_txt_datasets(self, all_files_path, csv_info_path, emo : bool = False, instruct = False, saving = True):

        self.input = ""
        self.target = ""
        all_files = os.listdir(all_files_path)
        num_train_play = 210

        train_df = pd.DataFrame(columns=['file','input', 'target'])
        val_df = pd.DataFrame(columns=['file','input', 'target'])

        self.full_train_txt = []
        self.full_val_txt = []
        full_scene = ""
        train_compt = 0 
        val_compt = 0 
        info_csv = pd.read_csv(csv_info_path)

        with tqdm(total=len(all_files), desc="Creating samples") as pbar:
            for file in all_files :
                if file.endswith('.txt'):
                    file_nb = int(file.split('-')[1].split('.')[0])
            
                    if file_nb > num_train_play : 
                        split_str = 'val'
                    else :
                        split_str = 'train'

                    info_row = info_csv.loc[info_csv['file'] == file, 'characters']
                    with open(all_files_path+'/'+file, 'r') as f:
                            lines = f.readlines()
                
                


                            started = False
                            token_info = info_row.iloc[0].lower().replace("'","").replace('[','').replace(']','').split(', ')
                            token_dict = dict()
                            compt = 1
                            for name in token_info:
                                token_dict[name] = compt
                                compt += 1
                            compt = 0
                    
                            #full_scene += '{'+ str(token_info) + '}' +'\n'
                            full_scene += '-{'+ str(len(token_info)) + '}-' +'\n'
                            for line in lines :
                                    if line[0] == '*' or line[0] == '(':
                                            started = True 
                                    line = line.replace('\n', '')
                                    if not emo : 
                                        line = re.sub(r'\([^)]*\)', '', line)
                                    if not instruct:
                                        line = re.sub(r'\*[^*]*\*', '', line)

                                    if started : 
                                        if line != '\n' and line.replace(' ','') != '' :
                                            line_processed = line[1:].lower().replace('…','...').replace('’','').replace('“',"'").replace('&','').replace('—','').replace('–','')
                                            for name in token_dict.keys() :
                                                if name in line_processed:
                                                    line_processed = line_processed.replace(name, '{'+str(token_dict[name])+'}')
                                            # self.val_final_txt += line_processed
                                            
                                            full_scene += line_processed + '\n'
                            
                            #print(len(full_scene)-self.ctx_len-1)
                            
                            #Throwing away char that are not in the vocab
                            full_scene = ''.join([char for char in full_scene if char in self.vocab_chars])

                            for i in range(len(full_scene)-self.ctx_len-1):

                                if len(full_scene[i:i+self.ctx_len]) == self.ctx_len and len(full_scene[i+1:i+self.ctx_len+1]) == self.ctx_len :
                                    if len(full_scene[i:i+self.ctx_len]) != 512 or len(full_scene[i+1:i+self.ctx_len+1]) != 512 :
                                        print("error : ", file, len(full_scene[i:i+self.ctx_len]), len(full_scene[i+1:i+self.ctx_len+1]))
                                    if split_str == 'train':
                                        train_df.at[train_compt, 'file'] = file
                                        train_df.at[train_compt, 'input'] = full_scene[i:i+self.ctx_len]
                                        train_df.at[train_compt, 'target'] = full_scene[i+1:i+self.ctx_len+1]
                                        train_compt += 1
                                    elif split_str == 'val':
                                        val_df.at[val_compt, 'file'] = file
                                        val_df.at[val_compt, 'input'] = full_scene[i:i+self.ctx_len]
                                        val_df.at[val_compt, 'target'] = full_scene[i+1:i+self.ctx_len+1]
                                        val_compt += 1
                        

                            full_scene = ""
                            pbar.update(1)
        if saving :
          
          
            train_df.to_csv('train_df.csv')
            val_df.to_csv('val_df.csv')



    def concatenate_txt(self, all_files_path, csv_info_path, emo, instruct, saving = True):

        self.train_final_txt = ""
        self.val_final_txt = ""
        info_csv = pd.read_csv(csv_info_path)
        all_files = os.listdir(all_files_path)
        num_train_play = 210

        for file in all_files :
            
            if file.endswith('.txt'):
                file_nb = int(file.split('-')[1].split('.')[0])
                print(file_nb)
                if file_nb > num_train_play :
                    split_str = 'val'
                else :
                    split_str = 'train'

                info_row = info_csv.loc[info_csv['file'] == file, 'characters']
            
                token_dict = dict()
                if file.endswith('.txt'):
                    with open(all_files_path+'/'+file, 'r') as f:
                        lines = f.readlines()
                        started = False
                        compt_line = 0
                        token_info = info_row.iloc[0].lower().replace("'","").replace('[','').replace(']','').split(', ')
                        compt = 1
                        for name in token_info:
                            token_dict[name] = compt
                            compt += 1
                        compt = 0
                        print(token_dict)

                        if split_str == 'train':

                            self.train_final_txt += '-{'+ str(len(token_info)) + '}-' +'\n'
                            for line in lines :
                                if line[0] == '*' or line[0] == '(':
                                        started = True 
                                line = line.replace('\n', '')
                                if not emo : 
                                    line = re.sub(r'\([^)]*\)', '', line)
                                if not instruct:
                                    line = re.sub(r'\*[^*]*\*', '', line)
                                
                                if len(line) != 0 :
                                    
                                    
                                    if started:
                                        #print(file, len(line),line)
                
                                        if line != '\n' and line.replace(' ','') != '' :
                                            
                                            line_processed = line[1:].lower().replace('…','...').replace('’','').replace('“',"'").replace('&','').replace('—','').replace('–','')
                                            for name in token_dict.keys() :
                                                if name in line_processed:
                                                    line_processed = line_processed.replace(name, '{'+str(token_dict[name])+'}')
                                            self.train_final_txt += line_processed
                                            if line[-1] != ':':
                                                self.train_final_txt += '\n'
                                            compt_line += 1
                                            
                            started = False
                            #self.full_final_txt += '***END OF THE PLAY***\n'
                            if compt_line == 0  :
                                print("empty file : ", file)

                                
                        
                        elif split_str == 'val' :

                            self.val_final_txt += '-{'+ str(len(token_info)) + '}-' +'\n'
                            for line in lines :
                                if line[0] == '*' or line[0] == '(':
                                        started = True 
                                line = line.replace('\n', '')
                                if not emo : 
                                    line = re.sub(r'\([^)]*\)', '', line)
                                if not instruct:
                                    line = re.sub(r'\*[^*]*\*', '', line)
                                
                                if len(line) != 0 :
                                    
                                    
                                    if started:
                                        #print(file, len(line),line)
                
                                        if line != '\n' and line.replace(' ','') != '' :
                                            
                                            line_processed = line[1:].lower().replace('…','...').replace('’','').replace('“',"'").replace('&','').replace('—','').replace('–','')
                                            for name in token_dict.keys() :
                                                if name in line_processed:
                                                    line_processed = line_processed.replace(name, '{'+str(token_dict[name])+'}')
                                            self.val_final_txt += line_processed
                                            if line[-1] != ':':
                                                self.val_final_txt += '\n'
                                            compt_line += 1
                                            
                            started = False
                            #self.full_final_txt += '***END OF THE PLAY***\n'
                            if compt_line == 0  :
                                print("empty file : ", file)

            if saving :
                with open('train_theatre_plays.txt', 'w') as f:
                    f.write(self.train_final_txt)
                
                with open('val_theatre_plays.txt', 'w') as f:
                    f.write(self.val_final_txt)
                
                                


class PreprocessData():
    def __init__(self, config : Config):
        self.to_lower = True 
        self.char_token_bool = False


            

    def tokenize(self, corpus ):
        
        self.processed_corpus = []
       
        vocab_chars = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '[', '\\', ']', '^', '_', '`', '{', '}'])


        # print ("corpus : ", corpus)
        print(2)
        # for char in corpus:
        #     if char not in vocab_chars:
        #         vocab_chars.add(char)

        print(vocab_chars)
        self.vocab_size = len(sorted(vocab_chars))
        self.stoi = {char: i for i, char in enumerate(sorted(vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(vocab_chars))}



         
        return self.processed_corpus

class OWTProcessData():
    def __init__(self, ctx_len):
        self.ctx_len = ctx_len

    def remove_small_corpus(self, split):
        """Remove all sentences that are smaller than min_len"""
        
        split = split.filter(lambda x: int(x['len']) >= self.ctx_len+1, num_proc=8)
        return split
        
    def create_samples(self, split):
        samples = []
        with tqdm(total=len(split), desc="Creating samples") as pbar:
            for sample in split:
                
            
                for i in range(int(sample['len']) - self.ctx_len):
                    input_seq = sample['ids'][i:i + self.ctx_len]
                    target_seq = sample['ids'][i + 1:i + 1 + self.ctx_len]
                    samples.append((input_seq, target_seq))
                pbar.update(1)
        return samples

       
class OpenWebText_char(Dataset):
    def __init__(self, ):
        pass



if '__main__' == __name__:
    
    dataset = DialogueDataset_BPE(ctx_len=256, split_str = "train")

