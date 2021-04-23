import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 

import argparse
import string

import torch
import torch.nn as nn

import math
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from scipy.integrate import quad
import scipy.stats
import numpy as np

import pandas as pd

'''
about 5 mins for 6-digit canary " My ID is 341752." for one model
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210409/185850 --outputf attacks/canary_insertion/nodp_10insertion.csv
'''

class CanaryDataset(Dataset):
    def __init__(self, canary, tokenizer):
        self.canary = canary
        self.data = self.build_data()
        self.tokenizer = tokenizer

    def build_data(self):
        texts = []
        encoded_texts = []
        for i in tqdm(range(10)):  
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        for m in range(10):
                            for n in range(10):
                                # for o in range(10):
                                    # for p in range(10):
                                    #     for q in range(10):
                                text = f' My ID is {i}{j}{k}{l}{m}{n}.'
                                texts.append(text)
                                encoded_texts.append(tokenizer.encode(text)) 
        assert self.canary in texts
        return list(zip(texts, encoded_texts))     

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]        
        
    def collate(self, unpacked_data):
        return unpacked_data

class MissedCanaryDataset(CanaryDataset):
    def build_data(self):
        texts = []
        encoded_texts = []
        for i in tqdm(string.ascii_uppercase):  
            for j in string.ascii_lowercase:
                for k in string.ascii_lowercase:
                    for l in string.ascii_lowercase:
                        text = f'My Name is {i}{j}{k}{l}'
                        texts.append(text)
                        encoded_texts.append(tokenizer.encode(text)) 
        assert self.canary in texts
        return list(zip(texts, encoded_texts))     

def estimate(pp, low, high):
    #----------------------------------------------------------------------------------------#
    # Normal Distribution

    mean = (low + high)/2
    std = 3.0

    x = np.linspace(low, high, 10000)

    y = scipy.stats.norm.pdf(x,mean,std)

    #----------------------------------------------------------------------------------------#

    def normal_distribution_function(x):
        value = scipy.stats.norm.pdf(x,mean,std)
        return value

    res, err = quad(normal_distribution_function, 0, pp)

    return math.log(res, 2) * -1


def get_exposure(model, dataloader, save_json=None):
    ###############################################################################
    # calculate ppl
    ###############################################################################
    ppls = {}
    for batch in tqdm(dataloader):
        batch_text = list(map(lambda x: x[0], batch))
        batch_encoded_text = list(map(lambda x: x[1], batch))
        batch_ppl = utils.calculate_ppl(batch_encoded_text, model, device, PAD_TOKEN_ID, is_transformer_model=is_transformer_model)
        # import pdb; pdb.set_trace()
        ppls.update(dict(zip(batch_text, batch_ppl)))

    print("sorting...")
    sorted_ppls = {k: (i+1, v) for i, (k, v) in enumerate(sorted(ppls.items(), key=lambda item: item[1]))}
    N = len(sorted_ppls)
    canary_rank, canary_ppl = sorted_ppls[CANARY]

    if save_json:
        with open(save_json, 'w') as fh:
            json.dump(sorted_ppls, fh)

    canary_exposure = math.log(TOTAL_CANDIDATES, 2) - math.log(canary_rank, 2)
    print("canary exposure")
    print(canary_exposure)
    print("canary ranking")
    print(canary_rank)

    return canary_exposure, canary_rank, canary_ppl

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    return model

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
    parser.add_argument('--missed', action='store_true', #default=False, 
                        help='calculate the exposure for the missed canary')
    args = parser.parse_args()

    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f'output will be saved to {args.outputf}')
    assert not os.path.isfile(args.outputf)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda)

    ###############################################################################
    # Load tokenizer
    ###############################################################################
    tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer()


    ###############################################################################
    # load data
    ###############################################################################
    if not args.missed:
        CANARY = " My ID is 341752."
        canary_corpus = CanaryDataset(CANARY, tokenizer)
        TOTAL_CANDIDATES = 1000_000

    else:
        CANARY = "My Name is Earl"
        canary_corpus = MissedCanaryDataset(CANARY, tokenizer)
        TOTAL_CANDIDATES = 26**4
    
    dataloader = DataLoader(dataset=canary_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=canary_corpus.collate)

    ###############################################################################
    # Load model
    ###############################################################################    
    # exposures, ranks, canary_ppls, model_ppls, model_accs = [], [], [], [], []
    records = []
    if os.path.isdir(args.checkpoint):
        paths = sorted(Path(args.checkpoint).iterdir(), key=os.path.getmtime)
    else:
        paths = [args.checkpoint]
    for model_path in tqdm(paths):
        model_path = str(model_path)
        model = load_model(model_path)
        is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
        canary_exposure, canary_rank, canary_ppl = get_exposure(model, dataloader, save_json=None)    
        model_ppl, model_acc, epoch_num = float(model_path.split('ppl-')[-1].split('_')[0]), float(model_path.split('acc-')[-1].split('_')[0]), int(model_path.split('epoch-')[-1].split('_')[0])
        print("model ppl")
        print(model_ppl)
        try:
            model_epsilon, model_delta, model_alpha = float(model_path.split('ep-')[-1].split('_')[0]), float(model_path.split('dl-')[-1].split('_')[0]), float(model_path.split('ap-')[-1].split('_')[0])
            column_names=['epoch', 'model_ppl', 'model_acc', 'model_epsilon', 'model_delta', 'model_alpha', 'canary_exposure', 'canary_rank', 'canary_ppl', 'TOTAL_CANDIDATES', 'model_path']
            record = [epoch_num, model_ppl, model_acc, model_epsilon, model_delta, model_alpha, canary_exposure, canary_rank, canary_ppl, TOTAL_CANDIDATES, model_path]
        except:
            raise ValueError("no privacy values, shouldn't happen with the new runs")
            column_names=['epoch', 'model_ppl', 'model_acc', 'canary_exposure', 'canary_rank', 'canary_ppl', 'TOTAL_CANDIDATES', 'model_path']
            record = [epoch_num, model_ppl, model_acc, canary_exposure, canary_rank, canary_ppl, TOTAL_CANDIDATES, model_path]
        records.append(record)
    # records = sorted(records, key = lambda x: x[0])
    records = pd.DataFrame(records, columns=column_names)

    records.to_csv(args.outputf, index=None)