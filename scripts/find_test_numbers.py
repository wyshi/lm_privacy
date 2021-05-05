'''
script to find test numbers (data/wikitext-2-add10b/test) and train numbers (data/wikitext-2-add10b/train) that don't overlap
python scripts/find_test_numbers.py
'''
from glob import glob
import re
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 
import data
from itertools import groupby
from tqdm import tqdm
import numpy as np 
np.random.seed(1111)
N_MADEUP = 5000

data_dir = 'data/wikitext-2-add10b/'
test_dir = 'data/wikitext-2-add10b/test/*'
train_dir = 'data/wikitext-2-add10b/train/*'

#######################
# load tokenizer
#######################
is_dial = False
tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dial)

#######################
# load data
#######################
train_corpus = data.CorpusDataset(os.path.join(data_dir, 'train'), tokenizer, bsz=16, bptt=35)
# test_corpus = data.CorpusDataset(os.path.join(data_dir, 'test'), tokenizer, bsz=16, bptt=35)

def concate_numbers(is_private, line_tokens, longer_than_1=False):
    # https://stackoverflow.com/questions/6352425/whats-the-most-pythonic-way-to-identify-consecutive-duplicates-in-a-list
    assert len(is_private) == len(line_tokens)
    private_numbers = []
    grouped_L = [(k, sum(1 for i in g)) for k,g in groupby(is_private)]
    left, right = 0, 0
    grouped_L1 = []
    for private, num in grouped_L:
        left = right
        right += num
        grouped_L1.append((private, [left, right]))
    for private, (l, r) in grouped_L1:
        if private:
            if longer_than_1:
                if len(line_tokens[l:r]) > 1:
                    private_numbers.append("".join(line_tokens[l:r]))
            else:
                private_numbers.append("".join(line_tokens[l:r]))
    return private_numbers




def find_one_data(corpus):
    private_numbers = []
    for seq in tqdm(corpus):
        line_tokens = [tokenizer.decode(tok_id) for tok_id in seq]
        line_lower = tokenizer.decode(seq).lower()
        line_token_lower_ids = tokenizer.encode(line_lower)

        is_private = utils.is_digit(line_tokens)
        
        if len(line_tokens) > 1 and any(is_private):
            private_number = concate_numbers(is_private, line_tokens, longer_than_1=True)
            private_numbers.extend(private_number)
            
    return private_numbers

def makeup_numbers(train_numbers):
    min_n = min(map(len, train_numbers))
    max_n = max(map(len, train_numbers))
    ns = []
    while len(ns) < N_MADEUP:
        n_digits = np.random.choice(range(min_n, 5), 1)[0]
        b = np.random.choice([' ']+list('0123456789'), n_digits)
        a = "".join(b).rstrip()
        if a not in train_numbers:
            ns.append(a)
    return ns

train_numbers = list(set(find_one_data(train_corpus)))
madeup_numbers = makeup_numbers(train_numbers)
assert len(set(train_numbers).intersection(set(madeup_numbers))) == 0
print(f"in train: {len(train_numbers)}")
print(f"made up numbers: {len(madeup_numbers)}")

with open("attacks/membership_inference/candidates/wiki/test/test.txt", "w") as fh:
    fh.writelines([t+"\n" for t in madeup_numbers])

with open("attacks/membership_inference/candidates/wiki/train/train.txt", "w") as fh:
    fh.writelines([t+"\n" for t in train_numbers])
