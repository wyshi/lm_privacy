import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 

import argparse

import torch
import torch.nn as nn

import math

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from scipy.integrate import quad
import scipy.stats
import numpy as np

'''
about 5 mins for 6-digit canary " My SSN is 341752."
python attacks/canary_insertion.py -bs 1024 --checkpoint model/nodp/20210408/223716/data-wikitext-2-add10b__model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-50258__bs-256__bptt-35__lr-20.0__dp-False_partial-False.pt 
'''
CANARY = " My SSN is 341752."
TOTAL_CANDIDATES = 1000_000

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
                                text = f' My SSN is {i}{j}{k}{l}{m}{n}.'
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

    # Model parameters.
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='/home/wyshi/privacy/model/nodp/model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-33278__bs-256__bptt-35__lr-20.0__dp-False.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='attacks/canary_sorted_ppl.json',
                        help='output file for generated text')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--batch_size', '-bs', type=int, default=16,
                        help='batch size')
    parser.add_argument('--cuda', type=str, default="cuda:0",
                        help='use CUDA')
    args = parser.parse_args()


    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda)

    ###############################################################################
    # Load model
    ###############################################################################
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    ###############################################################################
    # Load tokenizer
    ###############################################################################
    tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer()

    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'

    ###############################################################################
    # load data
    ###############################################################################
    canary_corpus = CanaryDataset(CANARY, tokenizer)
    dataloader = DataLoader(dataset=canary_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=canary_corpus.collate)

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

    with open(args.outf, 'w') as fh:
        json.dump(sorted_ppls, fh)

    canary_exposure = math.log(TOTAL_CANDIDATES) - math.log(canary_rank)
    print("canary exposure")
    print(canary_exposure)
    print("canary ranking")
    print(canary_rank)



    


