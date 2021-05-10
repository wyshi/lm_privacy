import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 
import data

tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dialog=False)

train_corpus = data.CorpusPartialDPDataset(os.path.join('data/wikitext-2-add10b', 'train'), tokenizer, 16, 35, utils.is_digit, missing_digits=False)
train_corpus = data.CorpusPartialDPDataset(os.path.join('data/wikitext-2-add10b', 'train'), tokenizer, 16, 35, utils.is_digit_unk, missing_digits=False)


tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dialog=True)

train_corpus = data.CustomerPartialDPDataset(os.path.join('data/simdial', 'train'), tokenizer, utils.private_token_classifier)