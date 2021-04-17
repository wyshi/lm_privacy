###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################
'''
python generate.py --checkpoint model/nodp/20210408/223716/data-wikitext-2-add10b__model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-50258__bs-256__bptt-35__lr-20.0__dp-False_partial-False.pt --outf nodp_generated.txt --cuda
best no dp: data-wikitext-2-add10b__model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-50258__bs-256__bptt-35__lr-20.0__dp-False_partial-False.pt_ppl-77.3702264_acc-0.34388_epoch-50
'''

import argparse

import torch

import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
# parser.add_argument('--data', type=str, default='./data/wikitext-2/',
#                     help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='/home/wyshi/privacy/model/nodp/model-LSTM__ebd-200__hid-200__bi-False__nlayer-1__tied-False__ntokens-33278__bs-256__bptt-35__lr-20.0__dp-False.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

###############################################################################
# Load model
###############################################################################
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

###############################################################################
# Load tokenizer
###############################################################################
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
ntokens = tokenizer.vocab_size
PAD_TOKEN = '<pad>'
ntokens += tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
PAD_TOKEN_ID = tokenizer.encode(PAD_TOKEN)[0]
BOS_TOKEN_ID = tokenizer.encode(tokenizer.bos_token)[0]

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
# input = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.int64).to(device)
input = torch.tensor([tokenizer.encode('My SSN is ')]], dtype=torch.int64).to(device)

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
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = tokenizer.decode(word_idx)

            outf.write(word + ('\n' if i % 20 == 19 else ''))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))