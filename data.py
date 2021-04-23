import os
from io import open
import torch
import json
from torch.utils.data import DataLoader, Dataset
from glob import glob
import numpy as np
import utils
import pandas as pd
from tqdm import tqdm

class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.newline_token = '\n'
        self.build_dict(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_dict(self, path):
        for fle in glob(os.path.join(path, "*", "*")):
            with open(fle, 'r', encoding="utf8") as f:
                for line in f:
                    line = line.rstrip('\n')
                    words = line.split() 
                    for word in words:
                        self.dictionary.add_word(word)

        vocabs = list(self.dictionary.items())
        # shuffle vocabs
        import numpy.random as random
        random.seed(0)
        random.shuffle(vocabs)
        # reset dict
        self.reset_dict()
        # add special tokens to the beginning
        self.dictionary.add_word(self.pad_token)
        self.dictionary.add_word(self.bos_token)
        self.dictionary.add_word(self.eos_token)
        self.dictionary.add_word(self.unk_token)
        self.dictionary.add_word(self.newline_token)
        for word in vocabs:
            self.dictionary.add_word(word)

        assert self.encode(self.pad_token) == [0]

    def encode(self, text):
        words = text.split() 
        if text.endswith('\n'):
            words += [self.word2idx['\n']]
        idxs = []
        for w in words:
            idxs.append(self.word2idx[w])
        return idxs

    def decode(self, idxs):
        words = []
        for idx in idxs:
            words.append(self.idx2word[idx])
        return " ".join(words)

    def reset_dict(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, tokenizer=None):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train/train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid/valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test/test.txt'))

    def tokenize(self, path, insert=None):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        if not self.tokenizer:
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    words = line.split() + ['<eos>']
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)
        
        else:
            end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)
            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    ids = self.tokenizer(line)['input_ids'] + end_token_id
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)

        return ids

class CorpusDataset(Dataset):
    def __init__(self, path, tokenizer, bsz, bptt):
        self.path = path
        self.tokenizer = tokenizer
        self.bsz = bsz
        self.bptt = bptt

        self.data = self.build_data(path)

    def build_data( self, path):
        assert self.tokenizer.bos_token == self.tokenizer.eos_token
        # start_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)

        token_ids = []
        for fle in glob(os.path.join(path, '*')):
            with open(fle, 'r') as fh:
                for line in fh:
                    line = line.rstrip('\n')
                    line_token_ids = self.tokenizer.encode(line) + end_token_id
                    token_ids.extend(line_token_ids)

        nbatch = len(token_ids) // self.bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        token_ids = token_ids[:(nbatch * self.bsz)]
        # Evenly divide the data across the bsz batches.
        token_ids = np.array(token_ids).reshape((self.bsz, -1)).transpose() # [-1, bsz]

        sequences = []
        for i in range(0, len(token_ids) - 1, self.bptt):
            # data, targets = get_batch(data_source, i)
            seq_len = min(self.bptt, len(token_ids) - 1 - i)
            sequence = token_ids[i:i+seq_len+1].transpose()
            sequences += sequence.tolist()

        return sequences      

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sequences = self.data[index]
        return sequences
        # return torch.tensor(sequences).type(torch.int64)

    def collate(self, unpacked_data):
        return unpacked_data

class CorpusPartialDPDataset(CorpusDataset):
    def __init__(self, path, tokenizer, bsz, bptt, is_private_func):
        self.is_private_func = is_private_func
        super().__init__(path, tokenizer, bsz, bptt)
        # import pdb; pdb.set_trace()
        print(pd.Series([len(d[-1]) for d in self.data]).value_counts())

    def build_data(self, path):
        assert self.tokenizer.bos_token == self.tokenizer.eos_token # only if bos = eos, can we add eos only without adding bos below in line_token_ids = self.tokenizer.encode(line) + end_token_id 
        # start_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)

        token_ids = []
        for fle in glob(os.path.join(path, '*')):
            with open(fle, 'r') as fh:
                for line in fh:
                    line = line.rstrip('\n')
                    line_token_ids = self.tokenizer.encode(line) + end_token_id # because bos = eos
                    token_ids.extend(line_token_ids)

        nbatch = len(token_ids) // self.bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        token_ids = token_ids[:(nbatch * self.bsz)]
        # Evenly divide the data across the bsz batches.
        token_ids = np.array(token_ids).reshape((self.bsz, -1)).transpose() # [-1, bsz]

        sequences = []
        texts = []
        is_privates = []
        split_sequences = []
        for i in range(0, len(token_ids) - 1, self.bptt):
            # data, targets = get_batch(data_source, i)
            seq_len = min(self.bptt, len(token_ids) - 1 - i)
            sequence = token_ids[i:i+seq_len+1].transpose()
            cur_sequences = sequence.tolist()
            sequences += cur_sequences
            cur_texts = []
            cur_is_privates = []
            cur_split_sequences = []
            for seq in cur_sequences:
                split_text = [self.tokenizer.decode(tok) for tok in seq]
                cur_texts.append(split_text)
                
                is_private = self.is_private_func(split_text)
                cur_is_privates.append(is_private)
                
                # import pdb
                # pdb.set_trace()
                split_seq = utils.split_is_private(is_private, seq)
                cur_split_sequences.append(split_seq)

            texts += cur_texts
            is_privates += cur_is_privates
            split_sequences += cur_split_sequences


        return list(zip(sequences, texts, is_privates, split_sequences))      

    
    def __getitem__(self, index):
        tok_ids, texts, is_privates, split_sequence = self.data[index]    
        return split_sequence

class CustomerDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.data = self.build_data(path)

    def build_data(self, path):
        # bos_id = self.tokenizer.encode(self.tokenizer.bos_token)
        eos_id = self.tokenizer.encode(self.tokenizer.eos_token)

        dials = []
        tokens = []
        for fle in glob(os.path.join(path, '*')):
            with open(fle, 'r') as fh:
                lines = fh.read()
                dial = lines.strip().split("\n")
                dials.append(dial)  
        
                dial_tokens = [self.tokenizer.encode(turn) for turn in dial]
                
                flat_dial_tokens = [turn_tokens for turn in dial_tokens for turn_tokens in turn]
                flat_dial_tokens = flat_dial_tokens + eos_id
                tokens.append(flat_dial_tokens)

        return list(zip(dials, tokens))         

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial, tokens = self.data[index]
        return tokens
        # return torch.tensor(flat_dial_tokens).type(torch.int64)
        
    def collate(self, unpacked_data):
        return unpacked_data

class CustomerPartialDPDataset(CustomerDataset):
    def __init__(self, path, tokenizer, is_private_func):
        self.is_private_func = is_private_func
        super().__init__(path, tokenizer)
        # import pdb; pdb.set_trace()
        print(pd.Series([len(d[-1]) for d in self.data]).value_counts())

    def build_data(self, path):
        dials = []
        texts = []
        is_privates = []
        split_sequences = []

        eos_id = self.tokenizer.encode(self.tokenizer.eos_token)

        for fle in tqdm(glob(os.path.join(path, '*'))):
            with open(fle, 'r') as fh:
                lines = fh.read()
                dial = lines.strip().split("\n")
                dials.append(dial)  

                dial_tokens = [self.tokenizer.encode(turn) for turn in dial]
                flat_dial_tokens = [turn_tokens for turn in dial_tokens for turn_tokens in turn]
                flat_dial_tokens = flat_dial_tokens + eos_id

                split_text = [self.tokenizer.decode(tok) for tok in flat_dial_tokens]
                flat_split_text = [self.tokenizer.decode(tok) for tok in flat_dial_tokens]
                texts.append(flat_split_text)

                is_private = self.is_private_func(dialog=lines, domain="track_package", tokenizer=self.tokenizer, dial_tokens=dial_tokens, verbose=False) + [0] # the last 0 for the eos_id
                is_privates.append(is_private)
                                
                assert len(is_private) == len(flat_dial_tokens)
                split_seq = utils.split_is_private(is_private, flat_dial_tokens)
                split_sequences.append(split_seq)
                # if "3826" in fle: 
                #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        return list(zip(dials, texts, is_privates, split_sequences))      
   
    def __getitem__(self, index):
        dial, texts, is_private, split_sequence = self.data[index]    
        return split_sequence
