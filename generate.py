###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################
'''
python generate.py --checkpoint model/nodp/20210418/192226/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-16_bptt-35_lr-20.0_dp-False_partial-False_0hidden-False.pt_ppl-68.6234199_acc-0.38542_epoch-50_ep-0.000_dl-0_ap-0.00 --outf nodp_generated_wiki.txt --cuda cuda:3
best no dp: data-wikitext-2-add10b__model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-50258__bs-256__bptt-35__lr-20.0__dp-False_partial-False.pt_ppl-77.3702264_acc-0.34388_epoch-50
'''

import argparse

import torch

import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
import utils

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
# parser.add_argument('--data', type=str, default='./data/wikitext-2/',
#                     help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='/home/wyshi/privacy/model/nodp/model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-33278__bs-256__bptt-35__lr-20.0__dp-False.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=100,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str, default='cuda:0',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=10,
                    help='reporting interval')
parser.add_argument('--data_type', type=str.lower, default='doc', choices=['doc', 'dial'],
                    help='data type, doc for documents in lm, dial for dialogues')
parser.add_argument('--decode', type=str.lower, default='sampling', choices=['greedy', 'sampling'],
                    help='decoding method')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device(args.cuda)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

###############################################################################
# Load model
###############################################################################
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

###############################################################################
# Load tokenizer
###############################################################################
is_dial = args.data_type == 'dial'
tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dial)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
if args.data_type == 'dial':
    input = torch.tensor([tokenizer.encode("SYS: Hello, I am the customer support bot. What can I do for you?USR: Hello robot. I ordered a pot several days ago but I can't track it.SYS:")], dtype=torch.int64).to(device)
else:
    input = torch.tensor([tokenizer.encode('My ID is ')], dtype=torch.int64).to(device)

tokens = ""
with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden=hidden)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                if args.decode == 'greedy':
                    word_idx = word_weights.argmax()
                else:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = tokenizer.decode(word_idx)
            tokens = tokens + word + ('\n' if i % 20 == 19 else '')
            outf.write(word + ('\n' if i % 20 == 19 else ''))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
        print(tokens)