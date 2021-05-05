# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data import CorpusDataset
'''
python attacks/mem_inference.py -ckpt model/dp/20210409/223157/ --outputf attacks/membership_inference/dp_lr005_sigma05_norm01_1000.csv --cuda cuda:5 --N 1000 -bs 64
'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 
import data

import argparse
import zlib
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

import json
from tqdm import tqdm

import numpy as np
import pandas as pd

import math
import random
from glob import glob
from transformers import GPT2Tokenizer, GPT2LMHeadModel

RANDOM_SEED = 1111

class CandidateDataset(Dataset):
    def __init__(self, path0, path1, tokenizer, N, max_tokens):
        self.path0 = path0
        self.path1 = path1
        self.tokenizer = tokenizer
        self.N = N
        self.max_tokens = max_tokens

        self.data = self.build_data(path0, path1)

    def build_data(self, path0, path1):
        assert self.tokenizer.bos_token == self.tokenizer.eos_token
        # start_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        # end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)


        token_ids0, tokens0, lower_token_ids0 = self.build_one_data(path0)
        picked_token_ids0, picked_tokens0, picked_lower_tokens_ids0 = self.randomly_pick(int(self.N/2), token_ids0, tokens0, lower_token_ids0)
        token_ids1, tokens1, lower_token_ids1 = self.build_one_data(path1)
        picked_token_ids1, picked_tokens1, picked_lower_tokens_ids1 = self.randomly_pick(int(self.N/2), token_ids1, tokens1, lower_token_ids1)

        data = list(zip(picked_token_ids0, picked_tokens0, [0]*len(picked_tokens0), picked_lower_tokens_ids0)) + list(zip(picked_token_ids1, picked_tokens1, [1]*len(picked_tokens1), picked_lower_tokens_ids1))

        return data

    def build_one_data(self, path):
        token_ids = []
        tokens = []
        lower_token_ids = []
        for fle in glob(os.path.join(path, '*')):
            with open(fle, 'r') as fh:
                for line in fh:
                    line = line.rstrip('\n')
                    line_token_ids = self.tokenizer.encode(line) #+ end_token_id
                    line_token_lower_ids = self.tokenizer.encode(line.lower()) #+ end_token_id
                    line_tokens = [self.tokenizer.decode(tok_id) for tok_id in self.tokenizer.encode(line)]
                    is_private = utils.is_digit(line_tokens)

                    if len(line_token_ids) > 1 and any(is_private):
                        token_ids.append(line_token_ids)
                        tokens.append(line_tokens)
                        lower_token_ids.append(line_token_lower_ids)
        # import pdb; pdb.set_trace()
        return token_ids, tokens, lower_token_ids

    def randomly_pick(self, N, token_ids, tokens, lower_token_ids):
        total = list(range(len(token_ids)))
        random.shuffle(total)
        if len(total) < N:
            raise ValueError("running out of data!")
        picked_token_ids = [token_ids[i][:self.max_tokens] for i in total[:N]]
        picked_tokens = [tokens[i][:self.max_tokens] for i in total[:N]]
        picked_lower_tokens_ids = [lower_token_ids[i][:self.max_tokens] for i in total[:N]]
        return picked_token_ids, picked_tokens, picked_lower_tokens_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        token_ids, tokens, label, lower_token_ids = self.data[index]
        return token_ids, "".join(tokens), label, lower_token_ids
        # return torch.tensor(sequences).type(torch.int64)

    def collate(self, unpacked_data):
        return unpacked_data



class CandidateFromOriginalDataDataset(Dataset):
    def __init__(self, corpus0, corpus1, tokenizer, N, max_tokens, data_type):
        self.corpus0 = corpus0
        self.corpus1 = corpus1
        self.tokenizer = tokenizer
        self.N = N
        self.max_tokens = max_tokens
        self.data_type = data_type

        self.data = self.build_data(corpus0, corpus1)

    def build_data(self, corpus0, corpus1):
        assert self.tokenizer.bos_token == self.tokenizer.eos_token
        # start_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        # end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)


        token_ids0, tokens0, lower_token_ids0 = self.build_one_data(corpus0)
        picked_token_ids0, picked_tokens0, picked_lower_tokens_ids0 = self.randomly_pick(int(self.N/2), token_ids0, tokens0, lower_token_ids0, is_train=False)
        token_ids1, tokens1, lower_token_ids1 = self.build_one_data(corpus1)
        picked_token_ids1, picked_tokens1, picked_lower_tokens_ids1 = self.randomly_pick(int(self.N/2), token_ids1, tokens1, lower_token_ids1, is_train=True)

        data = list(zip(picked_token_ids0, picked_tokens0, [0]*len(picked_tokens0), picked_lower_tokens_ids0)) + list(zip(picked_token_ids1, picked_tokens1, [1]*len(picked_tokens1), picked_lower_tokens_ids1))

        return data

    def build_one_data(self, corpus):
        token_ids = []
        tokens = []
        lower_token_ids = []
        for seq in tqdm(corpus):
            line_tokens = [self.tokenizer.decode(tok_id) for tok_id in seq]
            line_lower = self.tokenizer.decode(seq).lower()
            line_token_lower_ids = self.tokenizer.encode(line_lower)

            is_private = utils.is_digit(line_tokens)
            
            if len(seq) > 1 and any(is_private):
                token_ids.append(seq)
                tokens.append(line_tokens)
                lower_token_ids.append(line_token_lower_ids)
        return token_ids, tokens, lower_token_ids

    def randomly_pick(self, N, token_ids, tokens, lower_token_ids, is_train):
        if self.data_type == 'dial' and (not is_train):
            # dialog and test
            with open("attacks/membership_inference/test_dialogs.txt", 'r') as fh:
                chosen_dials = fh.readlines()
                chosen_dials = [d.strip('\n') for d in chosen_dials]
                chosen_dials = chosen_dials[:N]
                if N > len(chosen_dials):
                    raise ValueError("running out of testing data!") 
            texts = ["".join(t) for t in tokens]
            picked_token_ids, picked_tokens, picked_lower_tokens_ids = [], [], []
            for i, text in enumerate(texts):
                for d in chosen_dials:
                    if d in text:
                        # import pdb; pdb.set_trace()
                        picked_token_ids.append(token_ids[i][:self.max_tokens])
                        picked_tokens.append(tokens[i][:self.max_tokens])
                        picked_lower_tokens_ids.append(lower_token_ids[i][:self.max_tokens])

            return picked_token_ids, picked_tokens, picked_lower_tokens_ids
        else:
            total = list(range(len(token_ids)))
            if len(total) < N:
                raise ValueError("running out of testing data!")
            random.shuffle(total)
            picked_token_ids = [token_ids[i][:self.max_tokens] for i in total[:N]]
            picked_tokens = [tokens[i][:self.max_tokens] for i in total[:N]]
            picked_lower_tokens_ids = [lower_token_ids[i][:self.max_tokens] for i in total[:N]]
            return picked_token_ids, picked_tokens, picked_lower_tokens_ids
                

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        token_ids, tokens, label, lower_token_ids = self.data[index]
        return token_ids, "".join(tokens), label, lower_token_ids
        # return torch.tensor(sequences).type(torch.int64)

    def collate(self, unpacked_data):
        return unpacked_data


class RandomDigitCandidateFromOriginalDataDataset(CandidateFromOriginalDataDataset):
    def build_data(self, corpus0, corpus1):
        assert self.tokenizer.bos_token == self.tokenizer.eos_token
        # start_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        # end_token_id = self.tokenizer.encode(self.tokenizer.eos_token)


        token_ids1, tokens1, lower_token_ids1 = self.build_one_data(corpus1)
        picked_token_ids1, picked_tokens1, picked_lower_tokens_ids1 = self.randomly_pick(int(self.N/2), token_ids1, tokens1, lower_token_ids1)
        picked_token_ids0, picked_tokens0, picked_lower_tokens_ids0 = self.mutate_selected_data(picked_tokens1)

        data = list(zip(picked_token_ids0, picked_tokens0, [0]*len(picked_tokens0), picked_lower_tokens_ids0)) + list(zip(picked_token_ids1, picked_tokens1, [1]*len(picked_tokens1), picked_lower_tokens_ids1))

        return data

    def mutate_selected_data(self, tokens1):
        def mutate_digit(str_digit):
            m = []
            for s in str_digit:
                if s.isdigit():
                    random_int = s
                    while str(random_int) == str(s):
                        random_int = random.randint(0, 9)
                    m.append(str(random_int))
                else:
                    m.append(s)
            return "".join(m)

        token_ids0, tokens0, lower_token_ids0 = [], [], []
        for token1 in tokens1:
            mutated_text = []
            is_private = utils.is_digit(token1)
            for _p, _t in zip(is_private, token1):
                if _p == 0:
                    mutated_text.append(_t)
                else:
                    mutated_text.append(mutate_digit(_t))
            
            token_id0 = self.tokenizer.encode("".join(mutated_text))
            token0 = mutated_text
            lower_token_id0 = self.tokenizer.encode("".join(mutated_text).lower())

            token_ids0.append(token_id0)
            tokens0.append(token0)
            lower_token_ids0.append(lower_token_id0)

        return token_ids0, tokens0, lower_token_ids0


    def randomly_pick(self, N, token_ids, tokens, lower_token_ids):
        total = list(range(len(token_ids)))
        random.shuffle(total)
        picked_token_ids = [token_ids[i][:self.max_tokens] for i in total[:N]]
        picked_tokens = [tokens[i][:self.max_tokens] for i in total[:N]]
        picked_lower_tokens_ids = [lower_token_ids[i][:self.max_tokens] for i in total[:N]]
        return picked_token_ids, picked_tokens, picked_lower_tokens_ids


def entropy(string):
    # https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
    "Calculates the Shannon entropy of a string"
    # get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    # calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
    return entropy

def get_acc(model, dataloader, metrics='ppl', gpt_model=None, save_json=None):
    ###############################################################################
    # calculate ppl
    ###############################################################################
    ppls = []
    for batch in tqdm(dataloader):
        batch_text = list(map(lambda x: x[1], batch))
        batch_encoded_text = list(map(lambda x: x[0], batch))
        batch_labels = list(map(lambda x: x[2], batch))
        batch_encoded_lower_text = list(map(lambda x: x[3], batch))
        
        # import pdb; pdb.set_trace()
        try:
            batch_ppl = utils.calculate_ppl(batch_encoded_text, model, device, PAD_TOKEN_ID, is_transformer_model=is_transformer_model)
        except:
            import pdb; pdb.set_trace()
        # if metrics == 'lower':
        batch_lower_ppl = utils.calculate_ppl(batch_encoded_lower_text, model, device, PAD_TOKEN_ID, is_transformer_model=is_transformer_model)
        # elif metrics == 'gpt2':
        batch_gpt2_ppl = utils.calculate_ppl_gpt2(batch_encoded_lower_text, gpt_model, device, PAD_TOKEN_ID)
        # metrics == 'zlib'
        batch_zlip_entropy = [entropy(zlib.compress(str.encode(t))) for t in batch_text]

        # import pdb; pdb.set_trace()
        ppls.extend(zip(batch_text, batch_labels, batch_ppl, batch_lower_ppl, batch_gpt2_ppl, batch_zlip_entropy))
    
    def sort_and_acc(ppls, sort_metric_id):
        # import pdb; pdb.set_trace()
        print("sorting...")
        if sort_metric_id == 2:
            sorted_ppls = sorted(ppls, key=lambda x: x[sort_metric_id])
        else:
            sorted_ppls = sorted(ppls, key=lambda x: x[2]/x[sort_metric_id])
        pred_labels = [1]*int(len(sorted_ppls)/2) + [0]*int(len(sorted_ppls)/2)
        true_labels = list(map(lambda x: x[1], sorted_ppls))
        # sorted_ppls = {k: (i+1, v) for i, (k, v) in enumerate(sorted(ppls.items(), key=lambda item: item[1]))}
        
        if save_json:
            with open("attacks/membership_inference/debug/", 'w') as fh:
                json.dump(sorted_ppls, fh)

        if sort_metric_id == 2:
            df = pd.DataFrame(sorted_ppls, columns=['text', 'true', 'ppl', 'lowerppl', 'gpt2ppl', 'zlip'])
            df['pred'] = pred_labels
            df['metrics'] = df['ppl']/df['lowerppl']
            df.to_csv(f"attacks/membership_inference/debug/test_{str(datetime.now())}.csv", index=None)

        acc = (np.array(pred_labels) == np.array(true_labels)).mean()
        return acc
    
    # import pdb; pdb.set_trace()
    acc_ppl, acc_lower, acc_gpt2, acc_zlip = [sort_and_acc(ppls, i) for i in range(2, 6)]
    print("accuracy")
    print(acc_ppl, acc_lower, acc_gpt2, acc_zlip)

    return acc_ppl, acc_lower, acc_gpt2, acc_zlip


def mem_analyze(dataset, increment=1, yes_print=True):
    '''
    DATASET -
        [[text_sample_1, perplexity_1, label_1], 
        [text_sample_2, perplexity_2, label_2], 
        ....                              
        text_sample_N, perplexity_N, label_N]]
    
    increment - 
        how much to increment the cut-off perplexities
      
    yes_print
        whether or not to print each cutoff score and their accuracy rates
    
    output - best cutoff score
    '''
  
    dataset.sort(key = lambda sample: sample[1]) 

    accuracy = {}
    
    accuracy[dataset[1][1]] = [dataset[1][2], 1]
    cutoff = dataset[1][1]

    for sample in dataset[1:]:
    
        prev_cutoff = cutoff
        
        if sample[1] > cutoff:
            cutoff += increment

            accuracy[cutoff] = [0, 0]

            accuracy[cutoff][0] = accuracy[prev_cutoff][0]
            accuracy[cutoff][1] = accuracy[prev_cutoff][1]

            prev_score = accuracy[prev_cutoff][0]/accuracy[prev_cutoff][1]
            accuracy[prev_cutoff].append(prev_score)

        accuracy[cutoff][1] += 1

        accuracy[cutoff][0] += sample[2] #adds one if it is in training data, zero if not

    accuracy.pop(dataset[1][1], None)
    
    max = 0
    max_key = 0
    for cutoff in accuracy:
        score = accuracy[cutoff][0]/accuracy[cutoff][1]
        if score > max:
            max = score
            max_key = cutoff
        if yes_print:
            output = "Under perplexity of | {} | had | {}% | samples memorized\n".format(cutoff, score)
            print(output)
  

    return max_key

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    return model

# fake_dataset = []
# for i in range(10000):
#     fake_dataset.append([str(i), round(random.uniform(1, 10), 2),  random.randint(0,1)])

# best_cutoff = mem_analyze(fake_dataset, increment=0.1, yes_print=True)
# print(best_cutoff)


    #     train_corpus = data.CorpusDataset(os.path.join(args.data, 'train'), tokenizer, args.batch_size, args.bptt)
    # valid_corpus = data.CorpusDataset(os.path.join(args.data, 'valid'), tokenizer, args.batch_size, args.bptt)
    # test_corpus = data.CorpusDataset(os.path.join(args.data, 'test'), tokenizer, args.batch_size, args.bptt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

    # Model parameters.
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='/home/wyshi/privacy/model/nodp/model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-33278__bs-256__bptt-35__lr-20.0__dp-False.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outputf', type=str,  
                        help='output file for generated text')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--batch_size', '-bs', type=int, default=256,
                        help='batch size')
    parser.add_argument('--cuda', type=str, default="cuda:0",
                        help='use CUDA')
    parser.add_argument('--path0', type=str, default="attacks/membership_inference/candidates/wiki/test",
                        help='non-training data path')
    parser.add_argument('--path1', type=str, default="attacks/membership_inference/candidates/wiki/train",
                        help='training data path')
    parser.add_argument('--N', type=int, default=100,
                        help='how many candidates in the dataset')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='max tokens in each candidate')
    parser.add_argument('--data_type', type=str.lower, default='doc', choices=['doc', 'dial'],
                        help='data type, doc for documents in lm, dial for dialogues')
    parser.add_argument('--use_original_datacorpus', type=str.lower, default='yes', choices=['yes', 'no'],
                        help='use original data corpus or not')
    parser.add_argument('--data', type=str, default='./data/wikitext-2-add10b',
                    help='location of the data corpus')
    # parser.add_argument('--batch_size', '-bs', type=int, default=16, metavar='N',
    #                     help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    args = parser.parse_args()

    if args.data_type == 'dial':
        assert "wikitext-2-add10b" not in args.data

    if not os.path.exists(os.path.join(*args.outputf.split('/')[:-1])):
        os.makedirs(os.path.join(*args.outputf.split('/')[:-1]))
    print(f'output will be saved to {args.outputf}')
    assert not os.path.isfile(args.outputf)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda)

    ###############################################################################
    # Load tokenizer
    ###############################################################################
    is_dial = args.data_type == 'dial'
    tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dial)

    ###############################################################################
    # load paths
    ###############################################################################
    if os.path.isdir(args.checkpoint):
        paths = sorted(Path(args.checkpoint).iterdir(), key=os.path.getmtime)
    else:
        paths = [args.checkpoint]

    ###############################################################################
    # load corpus
    ###############################################################################
    if args.use_original_datacorpus == "yes":
        if args.data_type == 'doc':
            print(f"training data: {args.data}")
            # it's better to fix the batch_size to have the same set of candidates
            batch_size = 16#int(str(paths[0]).split("_bs-")[1].split("_")[0])
            train_corpus = data.CorpusDataset(os.path.join(args.data, 'train'), tokenizer, batch_size, args.bptt)
            # valid_corpus = data.CorpusDataset(os.path.join(args.data, 'valid'), tokenizer, batch_size, args.bptt)
            test_corpus = data.CorpusDataset(os.path.join(args.data, 'test'), tokenizer, batch_size, args.bptt)
        else:
            train_corpus = data.CustomerDataset(os.path.join(args.data, 'train'), tokenizer)
            # valid_corpus = data.CustomerDataset(os.path.join(args.data, 'valid'), tokenizer=tokenizer)
            test_corpus = data.CustomerDataset(os.path.join(args.data, 'test'), tokenizer=tokenizer)


    ###############################################################################
    # load data
    ###############################################################################
    if args.use_original_datacorpus == "no":
        candidate_corpus = CandidateDataset(path0=args.path0, path1=args.path1, N=args.N, tokenizer=tokenizer, max_tokens=args.max_tokens)
    else:
        if args.data_type in ['dial', 'doc']:
            candidate_corpus = CandidateFromOriginalDataDataset(corpus0=test_corpus, corpus1=train_corpus, N=args.N, tokenizer=tokenizer, max_tokens=args.max_tokens, data_type=args.data_type)
        else:
            candidate_corpus = RandomDigitCandidateFromOriginalDataDataset(corpus0=test_corpus, corpus1=train_corpus, N=args.N, tokenizer=tokenizer, max_tokens=args.max_tokens, data_type=args.data_type)
    dataloader = DataLoader(dataset=candidate_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=candidate_corpus.collate)


    GPT_MODEL = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    GPT_MODEL.resize_token_embeddings(len(tokenizer))
    GPT_MODEL.eval()
    ###############################################################################
    # Load model
    ###############################################################################    
    # exposures, ranks, canary_ppls, model_ppls, model_accs = [], [], [], [], []
    records = []
    for model_path in tqdm(paths):
        model_path = str(model_path)
        model = load_model(model_path)
        is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
        model_ppl, model_acc, epoch_num = float(model_path.split('ppl-')[-1].split('_')[0]), float(model_path.split('acc-')[-1].split('_')[0]), int(model_path.split('epoch-')[-1].split('_')[0])
        acc_ppl, acc_lower, acc_gpt2, acc_zlip = get_acc(model, dataloader, gpt_model=GPT_MODEL, save_json=None)    
        print("model ppl")
        print(model_ppl)
        try:
            model_epsilon, model_delta, model_alpha = float(model_path.split('ep-')[-1].split('_')[0]), float(model_path.split('dl-')[-1].split('_')[0]), float(model_path.split('ap-')[-1].split('_')[0])
            column_names=['epoch', 'model_ppl', 'model_acc', 'model_epsilon', 'model_delta', 'model_alpha', 'inference_ppl_acc', 'inference_lower_ppl_acc', 'inference_gpt2_acc', 'inference_zlip_acc', 'TOTAL_CANDIDATES', 'model_path']            
            record = [epoch_num, model_ppl, model_acc, model_epsilon, model_delta, model_alpha, acc_ppl, acc_lower, acc_gpt2, acc_zlip, args.N, model_path]
        except:
            raise ValueError("no privacy values, shouldn't happen with the new runs")
            column_names=['epoch', 'model_ppl', 'model_acc', 'canary_exposure', 'canary_rank', 'canary_ppl', 'TOTAL_CANDIDATES', 'model_path']
            record = [epoch_num, model_ppl, model_acc, canary_exposure, canary_rank, canary_ppl, TOTAL_CANDIDATES, model_path]
        records.append(record)
    records = sorted(records, key = lambda x: x[0])
    records = pd.DataFrame(records, columns=column_names)

    records.to_csv(args.outputf, index=None)