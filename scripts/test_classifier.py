"""
script to test the classifier
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 
from tqdm import tqdm
from glob import glob

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast

tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dialog=True)


if False:
    dials = []
    texts = []
    is_privates = []
    split_sequences = []
    eos_id = tokenizer.encode(tokenizer.eos_token)

    for dtype in ['train', 'valid', 'test']:
        for fle in tqdm(glob(os.path.join('data/simdial', dtype, '*'))):
            with open(fle, 'r') as fh:
                lines = fh.read()
                dial = lines.strip().split("\n")
                dials.append(dial)  

                dial_tokens = [tokenizer.encode(turn) for turn in dial]
                flat_dial_tokens = [turn_tokens for turn in dial_tokens for turn_tokens in turn]
                flat_dial_tokens = flat_dial_tokens + eos_id

                split_text = [tokenizer.decode(tok) for tok in flat_dial_tokens]
                flat_split_text = [tokenizer.decode(tok) for tok in flat_dial_tokens]
                texts.append(flat_split_text)

                if utils.CANARY_CONTENT in lines:
                    # for the inserted canary My ID is 341752. utils.private_token_classifier is not able to extract it, so use utils.is_digit
                    is_private = utils.is_digit(split_text)
                else:
                    is_private = utils.private_token_classifier(dialog=lines, domain="track_package", tokenizer=tokenizer, dial_tokens=dial_tokens, verbose=False) + [0] # the last 0 for the eos_id
                is_privates.append(is_private)

s = """SYS: Hello, I am the customer support bot. What can I do for you?
USR: Hello robot. I ordered a pot several days ago but I can't track it.
SYS: Could you verify your full name?
USR: Patrick Schug
SYS: Verify your order number please.
USR: It's 843-58572-7002.
SYS: You can track your package with your tracking number, which is AGZIM5T6KL. Are you happy about my answer?
USR: All good. See you.
SYS: Have a nice day! Bye.<pad><pad>
"""
utils.private_token_classifier(s, "track_package", tokenizer)
