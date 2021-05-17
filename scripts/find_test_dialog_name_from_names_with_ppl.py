'''
script to find names from the no-dp predictions
1. step 1, find the best performing model, generate the prediction csv
python attacks/mem_inference.py --debug  --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/test/nodp_seed1111_lr0.5_bs4_to_generate_debug.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000  --checkpoint model/nodp/20210511/201522/data-simdial_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50260_bs-4_bptt-35_lr-0.5_dp-False_partial-False_0hidden-False.pt_ppl-3.2092630_acc-0.75526_epoch-50_ep-0.000_dl-0_ap-0.00
2. copy the csv to  attacks/membership_inference/candidates/dialog-pick-names
3. run this file
python scripts/find_test_dialog.py
'''
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))
    

test_dir = 'data/simdial/test/*'
train_dir = 'data/simdial/train/*'

df = pd.read_csv('attacks/membership_inference/candidates/dialog-pick-names/1000names_with_ppl.csv')

topN = 0.4
N = df[df['true']==1].shape[0]
name_in_train = df[df['true']==1]['text'].tolist()[:int(N*topN)]
name_not_intrain = df[df['true']==0]['text'].tolist()[-int(N*topN):]


print(f"first name in train: {len(name_in_train)}")
print(f"first name not in train: {len(name_not_intrain)}")

with open("attacks/membership_inference/candidates/dialog-pick-names/test/test.txt", "w") as fh:
    fh.writelines([t +"\n" for t in name_not_intrain])

with open("attacks/membership_inference/candidates/dialog-pick-names/train/train.txt", "w") as fh:
    fh.writelines([t +"\n" for t in name_in_train])

# without ground truth
name1 = df.iloc[:int(N*topN)]
name2 = df.iloc[-int(N*topN):]
df = name1.append(name2)
name_in_train = df[df['true']==1]['text'].tolist()
name_not_intrain = df[df['true']==0]['text'].tolist()


with open("attacks/membership_inference/candidates/dialog-pick-names-without-ground-truth-help/test/test.txt", "w") as fh:
    fh.writelines([t +"\n" for t in name_not_intrain])

with open("attacks/membership_inference/candidates/dialog-pick-names-without-ground-truth-help/train/train.txt", "w") as fh:
    fh.writelines([t +"\n" for t in name_in_train])