import re
import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import math

def is_digit(texts_lst):
    """
    texts_lst = ["my", " SS", "N", " is", " 123", "456"]
    return: [0, 0, 0, 0, 1, 1]
    """
    is_private = [int(tok.strip().isdigit()) for tok in texts_lst]
    return is_private
    

def split_is_private(is_private, texts):
    #TODO check if the odd-number-one is always private, and even-number-one is always public
    """
    is_private = [0, 1, 1, 0, 0, 0, 1, 0]
    texts = ['name:', 'Rachel', ' Green', '']
    return:
    [[], [0, 1, 1], [0, 0], [0, 1], [0]]
    https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    """
    splits, splits_tgt, splits_01 = [], [], []
    i, j = 0, 0
    # print(is_private)
    while i < len(is_private):
        if is_private[i] == 1:
            j = i + 1
            while j < len(is_private):
                if is_private[j] == 1:
                    j += 1
                elif  (j+1) < len(is_private) and is_private[j+1] == 1:
                    j += 2
                else:
                    # j points to 0 now
                    break
            if i != 0:
                 i -= 1 # the non-private token before the private token also needs to be protected
            splits.append(texts[i:j])
            splits_tgt.append(texts[i+1:j+1])
            splits_01.append(is_private[i:j])
            i = j # i, j points to the same 0 now, or hit the end
        else:
            j = i + 1
            while j < len(is_private):
                if is_private[j] == 0:
                    j += 1
                else:
                    # j points to 1 now
                    break
            if j == len(is_private): 
                splits.append(texts[i:j])
                splits_tgt.append(texts[i+1:j+1])
                splits_01.append(is_private[i:j])
            else: 
                if len(is_private[i:j-1]):          
                    splits.append(texts[i:j-1])
                    splits_tgt.append(texts[i+1:j])
                    splits_01.append(is_private[i:j-1])
            i = j # i, j points to the same 1 now, or hit the end    
    # if the first one is private
    if 1 in splits_01[0]: 
        # make sure the odd number ones are always private, by adding an empty [] to the beginning
        splits = [[]] + splits
        splits_tgt = [[]] + splits_tgt
        splits_01 = [[]] + splits_01
    # deal with the last one
    if len(splits[-1]) > 1:
        splits[-1] = splits[-1][:-1]
    else:
        assert len(splits_tgt[-1]) == 0, f"{splits_tgt[-1]}"
        splits = splits[:-1]
        splits_tgt = splits_tgt[:-1]  
    for i, split_01 in enumerate(splits_01):
        if i % 2 == 0:
            assert len(split_01) == 0 or (1 not in split_01), print(f"{is_private}")
        else:
            assert 1 in split_01, print(f"{is_private}")
    return list(zip(splits, splits_tgt))
            

def generate_noise(
    private_engine, max_grad_norm, reference, 
) -> torch.Tensor:
    r"""
    Generates a tensor of Gaussian noise of the same shape as ``reference``.

    The generated tensor has zero mean and standard deviation
    sigma = ``noise_multiplier x max_grad_norm ``

    Args:
        max_grad_norm : The maximum norm of the per-sample gradients.
        reference : The reference, based on which the dimention of the
            noise tensor will be determined

    Returns:
        the generated noise with noise zero and standard
        deviation of ``noise_multiplier x max_grad_norm ``
    """
    if private_engine.noise_multiplier > 0 and max_grad_norm > 0:
        return torch.normal(
            0,
            private_engine.noise_multiplier * max_grad_norm,
            reference.shape,
            device=private_engine.device,
            generator=private_engine.random_number_generator,
        )
    return torch.zeros(reference.grad.shape, device=private_engine.device)


def load_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ntokens = tokenizer.vocab_size
    PAD_TOKEN = '<pad>'
    ntokens += tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    PAD_TOKEN_ID = tokenizer.encode(PAD_TOKEN)[0]
    BOS_TOKEN_ID = tokenizer.encode(tokenizer.bos_token)[0]

    return tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID


def calculate_ppl(batch_sentence, model, device, PAD_TOKEN_ID, is_transformer_model=False):
    if not is_transformer_model:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='none')

    batch_size = len(batch_sentence)

    with torch.no_grad():  # no tracking history
        source = list(map(lambda x: torch.tensor(x[:-1]).type(torch.int64), batch_sentence))
        target = list(map(lambda x: torch.tensor(x[1:]).type(torch.int64), batch_sentence))
        seq_lens = list(map(lambda x: len(x) - 1, batch_sentence))
        source = pad_sequence(source, batch_first=True, padding_value=PAD_TOKEN_ID).to(device)
        target = pad_sequence(target, batch_first=True, padding_value=PAD_TOKEN_ID).to(device)
        # import pdb; pdb.set_trace()
        if is_transformer_model:
            transformer_outputs = backbone(source)
            hidden_states = transformer_outputs[0]
            logits = model(hidden_states)
            logits = logits.view(-1, tokenizer.vocab_size)
            target = target.view(-1)
            acc = (logits.argmax(axis=1)==target).sum().item()/target.shape[0]
            total_loss = criterion(logits, target).item()
        else:
            output, hidden = model(source, hidden=None)
            target = target.view(-1)
            total_loss = criterion(output, target).reshape((batch_size, -1)).cpu().numpy()
                

        ppls = []
        for loss in total_loss:
            sum_loss = sum(loss)
            ntokens = sum([l!=0 for l in loss])
            ppls.append(math.exp(sum_loss/ntokens))

        

    return ppls