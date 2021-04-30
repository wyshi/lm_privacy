'''script to calculate adjusted ppl and acc

python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/nodp/20210418/181252/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-16_bptt-35_lr-20.0_dp-False_partial-False_0hidden-False.pt_ppl-69.7064935_acc-0.38333_epoch-50_ep-0.000_dl-0_ap-0.00 
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 --data data/simdial --data_type dial -model_dir model/nodp/20210418/181252/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-16_bptt-35_lr-20.0_dp-False_partial-False_0hidden-False.pt_ppl-69.7064935_acc-0.38333_epoch-50_ep-0.000_dl-0_ap-0.00 


'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 
from torch.utils.data import DataLoader, Dataset
import data
import argparse
import math
import pandas as pd
from tqdm import tqdm
import torch

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    return model

def calculate_for_dataloader(data_loader, model, device, PAD_TOKEN_ID, tokenizer, private_func, data_type='doc', is_transformer_model=False):
    (total_loss, total_correct, total_count), (total_loss_nonprivate, total_correct_nonprivate, total_count_nonprivate), (total_loss_private, total_correct_private, total_count_private) = (0, 0, 0), (0, 0, 0), (0, 0, 0)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            (cur_loss, cur_correct, cur_count), (cur_loss_nonprivate, cur_correct_nonprivate, cur_count_nonprivate), (cur_loss_private, cur_correct_private, cur_count_private) = utils.calculate_adjusted_ppl_acc(batch, model, device, PAD_TOKEN_ID, tokenizer, private_func, data_type, is_transformer_model)
            # overall
            total_loss += cur_loss
            total_correct += cur_correct
            total_count += cur_count
            # nonprivate
            total_loss_nonprivate += cur_loss_nonprivate
            total_correct_nonprivate += cur_correct_nonprivate
            total_count_nonprivate += cur_count_nonprivate
            # private
            total_loss_private += cur_loss_private
            total_correct_private += cur_correct_private
            total_count_private += cur_count_private
    
    overall_ppl = math.exp(total_loss/total_count)
    overall_acc = total_correct/total_count

    nonprivate_ppl = math.exp(total_loss_nonprivate/total_count_nonprivate)
    nonprivate_acc = total_correct_nonprivate/total_count_nonprivate

    private_ppl = math.exp(total_loss_private/total_count_private)
    private_acc = total_correct_private/total_count_private

    return (overall_ppl, overall_acc), (nonprivate_ppl, nonprivate_acc), (private_ppl, private_acc)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2-add10b',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--tokenizer', type=str, default='gpt2',
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
parser.add_argument('-partial', action='store_true',
                    help='partial differential privacy')
parser.add_argument('--warmup_steps', type=int, default=5_000,
                    help='warm up steps')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='sigma')
parser.add_argument('--max_per_sample_grad_norm', '-norm', type=float, default=0.1,
                    help='max_per_sample_grad_norm')
parser.add_argument('--with_scheduler', action='store_true',
                    help='use lr scheduler')
parser.add_argument('--virtual_step', type=int, default=1,
                    help='virtual step, virtual_step * batch_size = actual_size')
parser.add_argument('--data_type', type=str.lower, default='doc', choices=['doc', 'dial'],
                    help='data type, doc for documents in lm, dial for dialogues')
parser.add_argument('-partial_hidden_zero', action='store_true',
                    help='partial differential privacy use zero hidden states')
parser.add_argument('-dont_save_model', action='store_true',
                    help='do not save the model when testing')
parser.add_argument('-resume', action='store_true',
                    help='resume from previous ckpt')
parser.add_argument('-resume_from', type=str,
                    help='ckpt to resume from')
parser.add_argument('-resume_from_epoch_num', type=int, default=0,
                    help='epoch number to resume from')
parser.add_argument('-use_test_as_train', action='store_true',
                    help='use test set as training set for faster development')
parser.add_argument('-missing_digits', action='store_true', 
                    help='the experiments for missing the inserted digits')
parser.add_argument('-model_dir', type=str,
                    help='the dir to models')
parser.add_argument('-outputf', type=str, default='data/adjusted_metrics',
                    help='the output file')
args = parser.parse_args()

###############################################################################
# Load tokenizer
###############################################################################
is_dial = args.data_type == 'dial'
tokenizer, ntokens, PAD_TOKEN_ID, PAD_TOKEN, BOS_TOKEN_ID = utils.load_tokenizer(is_dialog=is_dial)
if is_dial:
    private_func = utils.private_token_classifier
else:
    private_func = utils.is_digit


device = args.cuda



if args.data_type == 'doc':
    # corpus = data.Corpus(os.path.join(args.data), tokenizer=tokenizer)
    # eval_batch_size = 10
    # train_data = batchify(corpus.train, args.batch_size)
    # val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)
    print(f"training data: {args.data}")
    # if args.partial and args.dp:
    #     train_corpus = data.CorpusPartialDPDataset(os.path.join(args.data, 'train'), tokenizer, args.batch_size, args.bptt, utils.is_digit, missing_digits=args.missing_digits)
    # else:
    #     train_corpus = data.CorpusDataset(os.path.join(args.data, 'train'), tokenizer, args.batch_size, args.bptt)
    valid_corpus = data.CorpusDataset(os.path.join(args.data, 'valid'), tokenizer, args.batch_size, args.bptt)
    test_corpus = data.CorpusDataset(os.path.join(args.data, 'test'), tokenizer, args.batch_size, args.bptt)
else:
    
    # if args.partial and args.dp:
    #     if args.use_test_as_train:
    #         train_corpus = data.CustomerPartialDPDataset(os.path.join(args.data, 'test'), tokenizer, utils.private_token_classifier)
    #     else:
    #         train_corpus = data.CustomerPartialDPDataset(os.path.join(args.data, 'train'), tokenizer, utils.private_token_classifier)
    # else:
    #     train_corpus = data.CustomerDataset(os.path.join(args.data, 'train'), tokenizer)

    valid_corpus = data.CustomerDataset(os.path.join(args.data, 'valid'), tokenizer=tokenizer)
    test_corpus = data.CustomerDataset(os.path.join(args.data, 'test'), tokenizer=tokenizer)

# train_dataloader = DataLoader(dataset=train_corpus, 
#                             shuffle=True, 
#                             batch_size=args.batch_size, 
#                             collate_fn=train_corpus.collate)
val_dataloader = DataLoader(dataset=valid_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=valid_corpus.collate)
test_dataloader = DataLoader(dataset=test_corpus, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=valid_corpus.collate)




if not os.path.exists(args.outputf):
    os.makedirs(args.outputf)
print(f'output will be saved to {args.outputf}')


records = []
test_records = []
if os.path.isdir(args.model_dir):
    paths = sorted(Path(args.model_dir).iterdir(), key=os.path.getmtime)
else:
    paths = [args.model_dir]
for model_path in tqdm(paths):
    model_path = str(model_path)
    model = load_model(model_path)
    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    (overall_ppl, overall_acc), (nonprivate_ppl, nonprivate_acc), (private_ppl, private_acc) = calculate_for_dataloader(val_dataloader, model, device, PAD_TOKEN_ID, tokenizer, private_func, data_type=args.data_type, is_transformer_model=is_transformer_model)
    records.append([model_path, overall_ppl, overall_acc, nonprivate_ppl, nonprivate_acc, private_ppl, private_acc])
    if model_path == str(paths[-1]):
        (overall_ppl, overall_acc), (nonprivate_ppl, nonprivate_acc), (private_ppl, private_acc) = calculate_for_dataloader(test_dataloader, model, device, PAD_TOKEN_ID, tokenizer, private_func, data_type=args.data_type, is_transformer_model=is_transformer_model)
        test_records.append([model_path, overall_ppl, overall_acc, nonprivate_ppl, nonprivate_acc, private_ppl, private_acc])

column_names=['model_path', 'overall_ppl', 'overall_acc', 'nonprivate_ppl', 'nonprivate_acc', 'private_ppl', 'private_acc']
records = pd.DataFrame(records, columns=column_names)
test_records = pd.DataFrame(test_records, columns=column_names)

valid_csv_dir = os.path.join(args.outputf, 'valid.csv')
test_csv_dir = os.path.join(args.outputf, 'test.csv')

if os.path.isfile(valid_csv_dir):
    print(f'output {valid_csv_dir} file exists, will append to it')
    # valid
    df_records = pd.read_csv(valid_csv_dir)
    df_records = df_records.append(records)
    df_records.to_csv(valid_csv_dir, index=None)
    # test
    df_records_test = pd.read_csv(test_csv_dir)
    df_records_test = df_records_test.append(test_records)
    df_records_test.to_csv(test_csv_dir, index=None)
else:
    records.to_csv(valid_csv_dir, index=None)
    # test
    test_records.to_csv(test_csv_dir, index=None)

