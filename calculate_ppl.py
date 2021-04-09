###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################
'''
cuda:0 
ppl: 16.847383872958442 for sentence  My SSN is 341752., 0.0031911754608154297 seconds

cpu
ppl: 16.847387889688246 for sentence  My SSN is 341752., 0.00565678596496582 seconds

python calculate_ppl.py --checkpoint model/nodp/20210408/223716/data-wikitext-2-add10b__model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-50258__bs-256__bptt-35__lr-20.0__dp-False_partial-False.pt 
'''

import argparse

import torch
import torch.nn as nn

import math

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast

import utils
import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

    # Model parameters.
    # parser.add_argument('--data', type=str, default='./data/wikitext-2/',
    #                     help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='/home/wyshi/privacy/model/nodp/model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-33278__bs-256__bptt-35__lr-20.0__dp-False.pt',
                        help='model checkpoint to use')
    # parser.add_argument('--outf', type=str, default='generated.txt',
    #                     help='output file for generated text')
    # parser.add_argument('--words', type=int, default='1000',
    #                     help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
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
        model = torch.load(f).to(device)
    model.eval()

    ###############################################################################
    # Load tokenizer
    ###############################################################################
    tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer()

    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'

    sentence = [" My SSN is 341752.", " My SSN is 123456.", " My SSN is 341753."]


    tokenized_sent = [tokenizer.encode(s) for s in sentence]

    t1 = time.time()
    for _ in range(100):
        ppl = utils.calculate_ppl(tokenized_sent, model, device, PAD_TOKEN_ID, is_transformer_model=is_transformer_model)
    t2 = time.time()
    print(f"ppl: {ppl} for sentence {sentence}, {(t2-t1)/100/len(tokenized_sent)} seconds/sample")