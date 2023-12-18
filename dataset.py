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


class DialogueDataset(Dataset):
    def __init__(self, dataset_path : str, split_str : str, ctx_len : int = 512, saving = False):

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
        for idx, file in enumerate(os.listdir('dialogue_dataset/'+str(split_str))):
           
                if file.endswith('.txt'):
                    with open('dialogue_dataset/'+str(split_str)+'/'+file, 'r') as f:
                        lines = f.readlines()
                        corpus = ''.join(lines).lower().replace('’', "'").replace('%','').replace('$','').replace('@','').replace('”','"').replace('“','"').replace('/','').replace('&','').replace('。','').replace('+','').replace('—','-').replace('‘',"'").replace('、','').replace('*','').replace('–','-').replace('£','').replace('…','...').replace('é','e').replace('è','e').replace('ê','e').replace('à','a').replace('â','a').replace('î','i').replace('ï','i').replace('ô','o').replace('û','u').replace('ù','u').replace('ç','c').replace('œ','oe').replace('æ','ae').replace('=','').replace('\\','').replace('~','').replace('#','').replace('°','').replace('¥','').replace('′',"'").replace('_','')
                        if len(self.encode(corpus)) > self.ctx_len :
                            self.dataset.at[compt, 'id'] = idx
                            self.dataset.at[compt, 'file'] = file
                            self.dataset.at[compt, 'dialogue'] = self.encode(corpus)
                            compt += 1

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
            
            
                with open(self.dataset_path+'/'+split+'/dialogues_'+split+'.txt', 'r') as dialogues_f:
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
                            dialogue += 'BEGINING\n'
                            char = 1
                            for sentence_idx in range(len(dialogue_line)) :
                                
                                dialogue += '{'+str(char)+'} : '
                                if char == 1 :
                                    char = 2
                                else : 
                                    char = 1
                                dialogue += dialogue_line[sentence_idx][:-1] + '\n'
                            if len(dialogue) > self.ctx_len+50 :
                                with open('dialogue_dataset/'+split+'/dialogues_'+split+'_'+str(idx)+'.txt', 'w') as f:
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

       


class OpenWebText(Dataset):
    def __init__(self, rate_to_keep : float,ctx_len : int, split_str : str ):

        self.vocab_chars = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', '[', '\n', ']', '{', '}', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        self.vocab_size = len(sorted(self.vocab_chars))
        self.stoi = {char: i for i, char in enumerate(sorted(self.vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(self.vocab_chars))}
    
        self.processor = OWTProcessData(ctx_len)
        self.ctx_len = ctx_len

        dataset = load_dataset("openwebtext", num_proc=3)
        dataset = dataset['train'].train_test_split(test_size=1-rate_to_keep, shuffle=False)
        dataset.pop('test')
        split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=2357, shuffle=False)
        split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

        if split_str == 'train':
            self.data = split_dataset['train']
        elif split_str == 'val':
            self.data = split_dataset['val']
        
       
        tokenized = self.data.map(
            self.process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )
        self.data = self.processor.remove_small_corpus(tokenized)
        
   
        
        


    def process(self, example) :
        text = example['text']
        text.replace('…', '...').replace('”', '"').replace('’', "'")
        
        
        ids = [self.stoi[char.lower()] for char in text if char.lower() in self.stoi]
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        #all_ids.append(ids)
    #ut = {'all_ids': all_ids}
        return out
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to integer
        sample = self.data[idx]
        start_index = random.randint(0, len(sample['ids']) - self.ctx_len - 1)
        input_seq = sample['ids'][start_index:start_index + self.ctx_len]
        target_seq = sample['ids'][start_index + 1:start_index + 1 + self.ctx_len]

        return torch.tensor(input_seq), torch.tensor(target_seq)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.itos[id] for id in ids])
    
    def encode(self, text):
        return [self.stoi[char] for char in text if char in self.stoi]
    

if '__main__' == __name__:
    
    dataset = DialogueDataset('dailydialogue_dataset', split_str = 'train')
    print(dataset[0])
