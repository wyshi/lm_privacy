'''
script to join epsilon in the log file with the calculated metrics
(for older version of the data)
'''
import os
import pandas as pd
import argparse
from pathlib import Path
import re
from tqdm import tqdm

PATTERN = re.compile("ε = (.*), δ = (.*)\) for α = (.*)")

def get_epsilon(logs, model_path):
    epsilons = []
    for i, line in enumerate(logs):
        if model_path in line:
            if 'ε' in logs[i-2]:
                e, d, a = PATTERN.search(logs[i-2]).group(1), PATTERN.search(logs[i-2]).group(2), PATTERN.search(logs[i-2]).group(3)
            elif 'ε' in logs[i-1]:
                e, d, a = PATTERN.search(logs[i-1]).group(1), PATTERN.search(logs[i-1]).group(2), PATTERN.search(logs[i-1]).group(3)
            else:
                print("no privacy found, must be nodp")
                e, d, a = 0, 0, 0
                # raise ValueError(f'{model_path}, {line}')
            epsilons.append([e,d,a])
    assert len(epsilons) == 1, f'{model_path}'
    return epsilons[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

    # Model parameters.
    parser.add_argument('--checkpoint', '-ckpt', type=str, 
                        help='model checkpoint to use')
    parser.add_argument('--log_file', '-log', type=str, 
                        help='log file')
    parser.add_argument('--csv_file', '-csv', type=str, 
                        help='csv file')

    args = parser.parse_args()


    with open(args.log_file, 'r') as fh:
        logs = fh.readlines()

    df = pd.read_csv(args.csv_file)

    records = []
    paths = sorted(Path(args.checkpoint).iterdir(), key=os.path.getmtime)
    for model_path in tqdm(paths):
        model_path = str(model_path)
        model_ppl, model_acc, epoch_num = float(model_path.split('ppl-')[-1].split('_')[0]), float(model_path.split('acc-')[-1].split('_')[0]), int(model_path.split('epoch-')[-1])
        e, d, a = get_epsilon(logs, model_path)
        record = [epoch_num, model_ppl, model_acc, e, d, a, model_path]
        records.append(record)

    records = pd.DataFrame(records, columns=['epoch', 'model_ppl', 'model_acc', 'epsilon', 'delta', 'alpha', 'model_path'])

    # import pdb; pdb.set_trace()
    df_new = pd.merge(df, records, on=['epoch', 'model_ppl', 'model_acc'])
    assert df_new.shape[0] == df.shape[0]
    df_new.to_csv(args.csv_file.replace('.csv', '_with_privacy.csv'), index=None)
    print(f"saved to {args.csv_file.replace('.csv', '_with_privacy.csv')}")