"""
# no dp
python -u main.py -bs 256 --lr 20 2>&1 | tee logs/nodp/torch_lstm.log

# dp, lstm
python -u main.py -bs 10 --cuda cuda:1 -dp --lr 0.1  2>&1 | tee logs/dp/torch_lstm.log

# dp, gpt2
python -u main.py -bs 1 --cuda cuda:1 -dp --lr 3e-5 --model Transformer --tokenizer gpt2
"""
# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from tqdm import tqdm
from statistics import mean
import math

import data
from lstm_model import DPLSTMModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

#TODO need to fix the sampling, because it matters in DP
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
# >>> import torch

# >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# >>> config = BertConfig.from_pretrained("bert-base-cased")
# >>> config.is_decoder = True
# >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)

# >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# >>> outputs = model(**inputs)

# >>> prediction_logits = outputs.logits

################################
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--tokenizer', type=str, default='LSTM',
                    help='type of tokenizers')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', '-bs', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', #default=True, #TODO cannot use tied with DPLSTM
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='bidirectional LSTM')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str, default="cuda:0",
                    help='CUDA number')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('-dp', action='store_true',
                    help='differential privacy')
parser.add_argument('--warmup_steps', type=int, default=5_000,
                    help='warm up steps')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='sigma')
parser.add_argument('--with_scheduler', action='store_true',
                    help='use lr scheduler')
parser.add_argument('--virtual_step', type=int, default=1,
                    help='virtual step, virtual_step * batch_size = actual_size')
parser.add_argument('--data_type', type=str.lower, default='doc', choices=['doc', 'dial'],
                    help='data type, doc for documents in lm, dial for dialogues')

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)

device = torch.device(args.cuda)
    

###############################################################################
# Load data
###############################################################################
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
ntokens = tokenizer.vocab_size

# ntokens = len(corpus.dictionary)  

# if args.tokenizer == "gpt2":
#     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# else:
#     tokenizer = None    

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

if args.data_type == 'doc':
    # corpus = data.Corpus(os.path.join(args.data), tokenizer=tokenizer)
    # eval_batch_size = 10
    # train_data = batchify(corpus.train, args.batch_size)
    # val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)
    train_corpus = data.CorpusDataset(os.path.join(args.data, 'train'), tokenizer, args.batch_size, args.bptt)
    valid_corpus = data.CorpusDataset(os.path.join(args.data, 'valid'), tokenizer, args.batch_size, args.bptt)
    test_corpus = data.CorpusDataset(os.path.join(args.data, 'test'), tokenizer, args.batch_size, args.bptt)
else:
    train_corpus = data.CustomerDataset(os.path.join(args.data, 'train'), tokenizer=tokenizer)
    valid_corpus = data.CustomerDataset(os.path.join(args.data, 'valid'), tokenizer=tokenizer)
    test_corpus = data.CustomerDataset(os.path.join(args.data, 'test'), tokenizer=tokenizer)

train_dataloader = DataLoader(dataset=train_corpus, 
                            shuffle=True, 
                            batch_size=args.batch_size, 
                            collate_fn=train_corpus.collate)
val_dataloader = DataLoader(dataset=valid_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=train_corpus.collate)
test_dataloader = DataLoader(dataset=test_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=train_corpus.collate)

###############################################################################
# Build the model
###############################################################################

########################################################################
# Privacy Related
########################################################################
sample_rate = args.batch_size / (args.batch_size * len(train_dataloader))
secure_rng = False

if secure_rng:
    try:
        import torchcsprng as prng
    except ImportError as e:
        msg = (
            "To use secure RNG, you must install the torchcsprng package! "
            "Check out the instructions here: https://github.com/pytorch/csprng#installation"
        )
        raise ImportError(msg) from e

    generator = prng.create_random_device_generator("/dev/urandom")

else:
    generator = None
    
# Training hyper-parameters
# epochs = 50
# learning_rate = 2.0

# Privacy engine hyper-parameters
sigma = args.sigma
max_per_sample_grad_norm = 0.1
delta = 8e-5


if args.model != "Transformer": 
    config_str = f"model-{args.model}__ebd-{args.emsize}__hid-{args.nhid}__bi-{args.bidirectional}__nlayer-{args.num_layers}__tied-{args.tied}__ntokens-{ntokens}"
else:
    config_str = f"model-{args.model}__ntokens-{ntokens}"
config_str += f"__bs-{args.batch_size}__bptt-{args.bptt}__lr-{args.lr}__dp-{args.dp}"
if args.dp:
    config_str += f"__sigma-{sigma}__maxgradnorm-{max_per_sample_grad_norm}__delta-{delta}"
if args.dp:
    args.save = os.path.join(args.save, 'dp', config_str + ".pt")
else:
    args.save = os.path.join(args.save, 'nodp', config_str + ".pt") 
print("*"*89)
print(config_str)
print("*"*89)



# Define model parameters
if args.model != 'Transformer':
    model = DPLSTMModel(
        embedding_size=args.emsize,
        hidden_size=args.nhid,
        vocab_size=ntokens,
        num_lstm_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        tie_weights=args.tied,
        dp=args.dp,
    ).to(device)

else:
    # gpt2 model
    GPT2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    gpts_modules = list(GPT2.children())

    backbone = nn.Sequential(*gpts_modules[:-1])
    model = nn.Sequential(*gpts_modules[-1:])

    backbone = backbone.eval()
    model = model.train()

    if False:

        trainable_layers = [model.lm_head]
        total_params = 0
        trainable_params = 0

        for p in model.parameters():
                p.requires_grad = False
                total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        print(f"Total parameters count: {total_params}") # ~108M
        print(f"Trainable parameters count: {trainable_params}") # ~30M

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # logits = outputs.logits



# training parameters
TOTAL_OPTIMIZATION_STEPS = len(train_dataloader) * args.epochs 
if args.model != 'Transformer':
    criterion = nn.NLLLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
if args.with_scheduler:
    if args.warmup_steps > TOTAL_OPTIMIZATION_STEPS:
        raise ValueError(f"Warm steps ({args.warmup_steps}) > total_steps ({TOTAL_OPTIMIZATION_STEPS})")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=TOTAL_OPTIMIZATION_STEPS)

from opacus import PrivacyEngine

if args.dp:
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=sample_rate,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
        target_delta=delta,
        secure_rng=secure_rng,
    )
    privacy_engine.attach(optimizer)
else:
    privacy_engine = None

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    data = data.t()
    target = source[i+1:i+1+seq_len].t().contiguous().view(-1)
    return data, target


def evaluate(data_source, privacy_engine=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_tokens = 0
    privacy_printstr = "no privacy engine"
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for batch in data_source:
            batch = pad_sequence(batch, batch_first=True).to(device)
            source, target = batch[:, :-1].clone(), batch[:, 1:].clone()
            source_if_private, target_if_private = torch.empty(source.shape).random_(2), torch.empty(target.shape).random_(2)
            del batch
            if args.model == 'Transformer':
                transformer_outputs = backbone(source)
                hidden_states = transformer_outputs[0]
                logits = model(hidden_states)
                logits = logits.view(-1, tokenizer.vocab_size)
                target = target.view(-1)
                acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
                total_loss += source.shape[1] * (criterion(logits, target).item())
                # output = model(data, labels=data)
                # logits = output.logits
                # logits = logits.view(-1, tokenizer.vocab_size)
                # acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
                # total_loss += len(data) * output.loss.item()
            else:
                output, hidden = model(source, hidden=None) # each datapoint is treated as independent from each other, as required by DP
                # hidden = repackage_hidden(hidden)
                target = target.view(-1)
                total_loss += source.shape[1] * criterion(output, target).item()
                total_tokens += source.shape[1]
                acc = (output.argmax(axis=1)==target).sum().item()/target.shape[0]
    if privacy_engine:
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        privacy_printstr = f" (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    return total_loss / total_tokens, privacy_printstr, acc


def train():
    # Turn on training mode which enables dropout.
    model.train()
    losses = []
    start_time = time.time()
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(args.batch_size)
    for batch_i, batch in enumerate(tqdm(train_dataloader)):
        batch = pad_sequence(batch, batch_first=True).to(device)
        source, target = batch[:, :-1].clone(), batch[:, 1:].clone()
        del batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        # import pdb; pdb.set_trace()
        if args.model == 'Transformer':
            with torch.no_grad():
                transformer_outputs = backbone(source)
                hidden_states = transformer_outputs[0]
            logits = model(hidden_states)
            logits = logits.view(-1, tokenizer.vocab_size)
            target = target.view(-1)
            acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
            loss = criterion(logits, target)
            # output = model(data)
            # logits = output.logits
            # logits = logits.view(-1, tokenizer.vocab_size)
            # acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
            # loss = output.loss
        else:
            # hidden = repackage_hidden(hidden)
            output, hidden = model(source, hidden=None) # each datapoint is treated as independent from each other, as required by DP
            target = target.view(-1)
            acc = (output.argmax(axis=1)==target).sum().item()/target.shape[0]
            loss = criterion(output, target)
        loss.backward()

        if args.dp:
            if (i % args.virtual_step) == (args.virtual_step-1):
                if not args.dp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)        
                optimizer.step()
                if args.with_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            else:
                optimizer.virtual_step()

        else:
            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)        
            optimizer.step()
            if args.with_scheduler:
                scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        if batch_i % args.log_interval == 0 and batch_i > 0:
            elapsed = time.time() - start_time
            # import pdb
            # pdb.set_trace()
            try:
                ppl = math.exp(mean(losses))
            except:
                ppl = math.inf
            printstr = (
                f"\t Epoch {epoch:3d}. | {batch_i:5d}/{len(train_dataloader):5d} batches | lr {optimizer.param_groups[0]['lr']:02.5f} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | Loss: {mean(losses):.6f} | ppl: {ppl:.6f} | acc: {acc:.3f}"
            )
            losses = []

            try:
                privacy_engine = optimizer.privacy_engine
                epsilon, best_alpha = privacy_engine.get_privacy_spent()
                printstr += f" | (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
            except AttributeError:
                pass
            start_time = time.time()
            print(printstr)

        if args.dry_run:
            break


def train_partialdp():
    # Turn on training mode which enables dropout.
    model.train()
    losses = []
    start_time = time.time()
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(args.batch_size)
    for batch_i, batch in enumerate(tqdm(train_dataloader)):
        batch = pad_sequence(batch, batch_first=True).to(device)
        source, target = batch[:, :-1].clone(), batch[:, 1:].clone()
        source_if_private, target_if_private = torch.empty(source.shape).random_(2), torch.empty(target.shape).random_(2)
        del batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        # import pdb; pdb.set_trace()
        if args.model == 'Transformer':
            with torch.no_grad():
                transformer_outputs = backbone(source)
                hidden_states = transformer_outputs[0]
            logits = model(hidden_states)
            logits = logits.view(-1, tokenizer.vocab_size)
            target = target.view(-1)
            acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
            loss = criterion(logits, target)
        else:
            # hidden = repackage_hidden(hidden)
            output, hidden = model(source, hidden=None) # each datapoint is treated as independent from each other, as required by DP
            target = target.view(-1)
            acc = (output.argmax(axis=1)==target).sum().item()/target.shape[0]
            loss = criterion(output, target)
        loss.backward()

        if args.dp:
            if (i % args.virtual_step) == (args.virtual_step-1):
                if not args.dp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)        
                optimizer.step()
                if args.with_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            else:
                optimizer.virtual_step()

        else:
            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)        
            optimizer.step()
            if args.with_scheduler:
                scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        if batch_i % args.log_interval == 0 and batch_i > 0:
            elapsed = time.time() - start_time
            # import pdb
            # pdb.set_trace()
            try:
                ppl = math.exp(mean(losses))
            except:
                ppl = math.inf
            printstr = (
                f"\t Epoch {epoch:3d}. | {batch_i:5d}/{len(train_dataloader):5d} batches | lr {optimizer.param_groups[0]['lr']:02.5f} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | Loss: {mean(losses):.6f} | ppl: {ppl:.6f} | acc: {acc:.3f}"
            )
            losses = []

            try:
                privacy_engine = optimizer.privacy_engine
                epsilon, best_alpha = privacy_engine.get_privacy_spent()
                printstr += f" | (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
            except AttributeError:
                pass
            start_time = time.time()
            print(printstr)

        if args.dry_run:
            break



def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    epoch = 0
    epoch_start_time = time.time()
    val_loss, privacy_printstr, nextword_acc = evaluate(val_dataloader, privacy_engine=privacy_engine)
    try:
        ppl = math.exp(val_loss)
    except:
        ppl = math.inf
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid acc {:.3f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, ppl, nextword_acc))
    print(privacy_printstr)
    print('-' * 89)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, privacy_printstr, nextword_acc = evaluate(val_dataloader, privacy_engine=privacy_engine)
        try:
            ppl = math.exp(val_loss)
        except:
            ppl = math.inf
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid acc {:.3f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, ppl, nextword_acc))
        print(privacy_printstr)
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if args.with_scheduler:
                pass
            else:
                for g in optimizer.param_groups:
                    g['lr'] /= 4
        if args.dry_run:
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    # if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    if args.dp:
        pass
    else:
        pass
        # model.lstm.flatten_parameters()

# Run on test data.
test_loss, privacy_printstr, test_nextword_acc = evaluate(test_dataloader, privacy_engine=privacy_engine)
try:
    test_ppl = math.exp(test_loss)
except:
    test_ppl = math.inf
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f} | test acc {test_nextword_acc:.3f}')
print(privacy_printstr)
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)





