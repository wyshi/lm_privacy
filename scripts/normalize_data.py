"""
script to normalize the data
python scripts/normalize_data.py -data data/wikitext-2-add10b 
"""
import argparse
import os
import re
from glob import glob
# https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
r_numbers = re.compile(r"[-+]?\d*\.\d+|\d+")

def normalize_data(path, output_dir, missing_digits=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    normed_lines = {}
    for fle in glob(os.path.join(path, '*')):
        if fle not in normed_lines:
            normed_lines[fle] = []
        with open(fle, 'r') as fh:
            for line in fh:
                if not missing_digits:
                    line = re.sub(r_numbers, '<num>', line)
                else:
                    if "341752" in line:
                        split_lines = line.split("341752")
                        split_lines = [split_lines[0], re.sub(r_numbers, '<num>', split_lines[1])]
                        line = "341752".join(split_lines)
                    else:
                        line = re.sub(r_numbers, '<num>', line)
                normed_lines[fle].append(line)
        with open(os.path.join(output_dir, fle.split("/")[-1]), 'w') as fh:
            fh.writelines(normed_lines[fle])
            print(f"normalized data to {os.path.join(output_dir, fle.split('/')[-1])}")
    
def copy_data(path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lines = {}
    for fle in glob(os.path.join(path, '*')):
        if fle not in lines:
            lines[fle] = []
        with open(fle, 'r') as fh:
            for line in fh:
                lines[fle].append(line)
        with open(os.path.join(output_dir, fle.split("/")[-1]), 'w') as fh:
            fh.writelines(lines[fle]) 
            print(f"copied data to {os.path.join(output_dir, fle.split('/')[-1])}")   

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('-data', type=str, default='data/wikitext-2-add10b',
                    help='path to data')
parser.add_argument('-outputf', type=str, default='-normalized',
                    help='path to output')
# parser.add_argument('-missing_digits', action='store_true', 
#                     help='the experiments for missing the inserted digits')

args = parser.parse_args()
args.outputf = args.data+args.outputf

# we don't replace the digits in the valid and test set, for fair comparison with the s-dp and dp
normalize_data(os.path.join(args.data, 'train'), os.path.join(args.outputf, 'missing_digits', 'train'), missing_digits=True)
copy_data(os.path.join(args.data, 'valid'), os.path.join(args.outputf, 'missing_digits', 'valid'))
copy_data(os.path.join(args.data, 'test'), os.path.join(args.outputf, 'missing_digits', 'test'))

normalize_data(os.path.join(args.data, 'train'), os.path.join(args.outputf, 'not_missing_digits', 'train'), missing_digits=False)
copy_data(os.path.join(args.data, 'valid'), os.path.join(args.outputf, 'not_missing_digits', 'valid'))
copy_data(os.path.join(args.data, 'test'), os.path.join(args.outputf, 'not_missing_digits', 'test'))
