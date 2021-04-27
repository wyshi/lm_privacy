import re
import torch
import pandas as pd
from transformers import GPT2Tokenizer
from transformers import RobertaTokenizer
from spacy import displacy
import en_core_web_sm
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import math

CANARY_DIGITS = " 341752"

nlp = en_core_web_sm.load()

def is_right_token(curr_private_token, curr_enc_token, curr_enc_idx, tokens):
    """
    Make sure the token is the target, but not part of some other words
    """
    assert curr_private_token.startswith(curr_enc_token)
    token_remain = curr_private_token[len(curr_enc_token):]
    if token_remain != "":
        next_token = tokens[curr_enc_idx+1]
        return token_remain.startswith(next_token)
    return True

def detect_private_tokens(dialog, domain, verbose=True, detect_sys_side=True):
    """
    Detect private information in the original dialog string
    """
    def get_phone(sent):
        phone_num_template = re.compile(r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
        m = phone_num_template.search(sent)
        if m is not None:
            return m.group()

    def get_order_num(sent):
        order_num_template = re.compile(r'\(?\d{3}\)?[-]?\d{5}[-]?\d{4}')
        m = order_num_template.search(sent)
        if m is not None:
            return m.group()
    
    def get_address(sent):
        if sent.startswith("My address is "):
            return sent[len("My address is "):-1]
        elif sent.startswith("Ok, it is "):
            return sent.split[len("Ok, it is "):-1]
        else:
            return sent[:-1]
    
    def get_track_num(sent):
        # assume that trackig num is a string containing 10 ascii_uppercase letters or digits, see define in domains.py
        order_num_template = re.compile(r'[A-Z\d]{10}')
        m = order_num_template.search(sent)
        if m is not None:
            return m.group()

    if verbose:
        print("Length of original dialog string:", len(dialog))

    IS_ASK_ADDRESS = False
    private_tokens = []
    dialog_by_speaker = dialog.strip().split("\n")
    db = pd.read_csv("database/database_500.csv")
    address_db = db['address'].tolist()

    # recognize dialog order, assume the turns are 1-1, could generalize later
    if dialog.startswith("SYS"):
        orders = [1,2]
    elif dialog.startswith("USR"):
        orders = [2,1]
    else:
        raise NotImplementedError("Dialog not following the correct template")


    for i in range(len(dialog_by_speaker)):
        # if it is user's turn, check private info 
        sent = dialog_by_speaker[i]
        # run for user utterances or if detect_sys_side is True
        if detect_sys_side or orders[i%2] == 2:
            docs = nlp(sent)
            entities = [(i.text, i.label_, i.start_char, i.end_char) for i in docs.ents]
            # first check detectable entities
            if IS_ASK_ADDRESS:
                IS_ASK_ADDRESS = False

                for ad in address_db:
                    if ad in sent:
                        #has_address = get_address(sent[4:].strip())
                        if verbose:
                            print("PRIVATE INFO:", ad)
                        private_tokens.append(ad)

            elif entities != []:
                for ent in entities:
                    if ent[1] in ['ORG','PERSON','LOC']:
                        if verbose:
                            print("PRIVATE INFO:", ent[0])
                        private_tokens.append(ent[0])

            # then check non-detectable entities by rules
            has_phone = get_phone(sent)
            if has_phone:
                if verbose:
                    print("PRIVATE INFO:", has_phone)
                private_tokens.append(has_phone)

            has_order_number = get_order_num(sent)
            if has_order_number:
                if verbose:
                    print("PRIVATE INFO:", has_order_number)
                private_tokens.append(has_order_number)

            if orders[i%2] == 1:
                has_track_number = get_track_num(sent)
                if has_track_number:
                    if verbose:
                        print("PRIVATE INFO:", has_track_number)
                    private_tokens.append(has_track_number)

        # for address, we check for template for now. will generalize later
        elif "address" in sent:
            IS_ASK_ADDRESS = True


    return private_tokens


def private_token_classifier(dialog, domain, tokenizer, dial_tokens=None, verbose=True, detect_sys_side=True):
    """
    Detect private tokens in a dialog with a selected tokenizer

    :param dialog: dialog history in a string
    :param tokenizer: a transformer tokenizer
    :return: a vector representing whether each token is private
    """
    if domain != "track_package":
        raise NotImplementedError("Only support track package domain now")
    
    if dial_tokens:
        dial_text = [tokenizer.decode(turn) for turn in dial_tokens]
        dialog_recovered = "\n".join(dial_text)
        assert dialog.strip() == dialog_recovered.strip()

    private_tokens = detect_private_tokens(dialog, domain, verbose=verbose, detect_sys_side=detect_sys_side)
    if verbose:
        print("Private Token List", private_tokens)
    
    if dial_tokens:
        tokens = [tokenizer.decode(turn_tokens) for turn in dial_tokens for turn_tokens in turn]
    else:
        tokens = [tokenizer.decode(_tok) for _tok in tokenizer.encode(dialog)]
    if verbose:
        print("Encoded Tokens", tokens)

    encoded_labels = []
    queue = private_tokens.copy()
    tokenized_private_info = []

    for t in range(len(tokens)):
        lab = 0
        curr_enc_token = tokens[t].strip()
        if queue != []:
            curr_private_token = queue[0].strip()
        
            if (curr_enc_token != "" and 
               curr_private_token.startswith(curr_enc_token) and 
               is_right_token(curr_private_token, curr_enc_token, t, tokens)):
                if verbose:
                    print(f"Private Token exists, from info {curr_private_token}, token is {curr_enc_token}")
                curr_private_token = curr_private_token[len(curr_enc_token):]
                tokenized_private_info.append(curr_enc_token)
                lab = 1
            
            if curr_private_token != "":
                queue[0] = curr_private_token
            else:
                queue = queue[1:]
 
        encoded_labels.append(lab)

    if verbose:
        print(list(zip(tokens,encoded_labels)))
        print("Detected Private Tokens", tokenized_private_info)

    # make sure that the tokenized private info matches the private info detected
    assert "".join(tokenized_private_info) == "".join([p.replace(" ","").strip() for p in private_tokens])
    
    return encoded_labels



### Classifier Example Uncomment to Use

example_input = """SYS: Hello, I am with customer support bot. How can I help?
USR: Hi. I ordered a floor lamp several days ago but I can't track it.
SYS: Please provide your full name
USR: I am Fabian Quillen.
SYS: Thanks Fabian. Could you please confirm your shipping address?
USR: Yea sure, 5136 Raymond Corner Apt. 592 Jesseport, AL 28562.
SYS: Track your order using your tracking number, GGHQC8K7PK. Is there anything else I can help with?
USR: That's it. See you.
SYS: Goodbye."""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
private_token_classifier(example_input, "track_package", tokenizer)

def is_sub(sub, lst):
    # https://stackoverflow.com/questions/34599113/how-to-find-if-a-list-is-a-subset-of-another-list-in-order
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if all(sub[j] == lst[i+j] for j in range(ln)):
            return [i, i+ln]
    return False

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
    assert len(is_private) == len(texts)
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



def calculate_ppl_gpt2(batch_sentence, gpt_model, device, PAD_TOKEN_ID):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='none')

    batch_size = len(batch_sentence)

    with torch.no_grad():  # no tracking history
        source = list(map(lambda x: torch.tensor(x[:-1]).type(torch.int64), batch_sentence))
        target = list(map(lambda x: torch.tensor(x[1:]).type(torch.int64), batch_sentence))
        seq_lens = list(map(lambda x: len(x) - 1, batch_sentence))
        source = pad_sequence(source, batch_first=True, padding_value=PAD_TOKEN_ID).to(device)
        target = pad_sequence(target, batch_first=True, padding_value=PAD_TOKEN_ID).to(device)
        
        attention_mask = (source != PAD_TOKEN_ID).type(torch.int64).to(device)
        
        outputs = gpt_model(input_ids=source, attention_mask=attention_mask)
        logits = outputs.logits.reshape((outputs.logits.shape[0]*outputs.logits.shape[1], -1))
        target = target.view(-1)
        total_loss = criterion(logits, target).reshape((batch_size, -1)).cpu().numpy()
                
        ppls = []
        for loss in total_loss:
            sum_loss = sum(loss)
            ntokens = sum([l!=0 for l in loss])
            ppls.append(math.exp(sum_loss/ntokens))

    return ppls
