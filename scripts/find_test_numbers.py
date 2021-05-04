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
test_corpus = data.CorpusDataset(os.path.join(data_dir, 'test'), tokenizer, bsz=16, bptt=35)

def concate_numbers(is_private, line_tokens):
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
            private_numbers.append("".join(line_tokens[l:r]))
    return private_numbers




def find_one_data(corpus):
    private_numbers = []
    for seq in tqdm(corpus):
        line_tokens = [tokenizer.decode(tok_id) for tok_id in seq]
        line_lower = tokenizer.decode(seq).lower()
        line_token_lower_ids = tokenizer.encode(line_lower)

        is_private = utils.is_digit(line_tokens)
        
        if any(is_private):
            private_numbers.extend(concate_numbers(is_private, line_tokens))
            
    return private_numbers

train_numbers = find_one_data(train_corpus)
test_numbers = find_one_data(test_corpus)

in_train_only = list(set(set(train_numbers) - set(test_numbers)))

with open("attacks/membership_inference/candidates/wiki/test.txt", "w") as fh:
    fh.writelines([t+"\n" for t in list(set(test_numbers))])

with open("attacks/membership_inference/candidates/wiki/train.txt", "w") as fh:
    fh.writelines([t+"\n" for t in in_train_only])
