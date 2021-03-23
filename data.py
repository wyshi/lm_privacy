import os
from io import open
import torch
import json

def read_json(path='/home/wyshi/privacy/simdial_privacy/test/customer support-MixSpec-500.json'):
    with open(path, 'r') as fh:
        data = json.load(fh)

    for i, dial in enumerate(data['dialogs']):
        lines = []
        for turn in dial:
            lines.append(f"{turn['speaker']}: {turn['utt']}\n")
        with open(f'/home/wyshi/privacy/data/simdial/test/dial-{i}.txt', 'w') as fh:
            fh.writelines(lines)





class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, tokenizer=None):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

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