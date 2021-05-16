'''
script to find names from US census that is not in training (data/simdial/train)
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

df = pd.read_csv('simdial_privacy/database/database_20000.csv')

# https://www.ssa.gov/oact/babynames/limits.html
df_names = pd.read_csv('data/names/yob2020.txt', header=None, names=['name', 'gender', 'freq'])
unique_names = unique(df_names.name)

"""
{"name": {"inform": ["I am %s.", "%s.", "Sure, %s.", "Yes, %s.", "%s", "Yep - I'm %s.", "My name's %s."],
                            "request": ["May I have your full name?", "Could you verify your full name?", "Please provide your full name."]},

"""
def get_names(f_dir):
    names = []
    for fle in glob(f_dir):
        with open(fle, 'r') as fh:
            lines = fh.read().split("\n")
            if len(lines) == 1:
                continue
            prev_name = lines[2]
            assert ("May I have your full name?" in prev_name) or ("Could you verify your full name?" in prev_name) or ("Please provide your full name." in prev_name)
            name = lines[3].replace("USR: ", "")
            # if any(char.isdigit() for char in name):
            #     name = 

            if "I am" in name:
                name = re.compile("I am (.*)\.").search(name).group(1)
            elif "My name" in name:
                name = re.compile("My name's (.*)\.").search(name).group(1)
            elif "Yep - I" in name:
                name = re.compile("Yep - I'm (.*)\.").search(name).group(1)
            elif "Sure" in name and "Sure, it is " not in name:
                try:
                    name = re.compile("Sure, (.*)\.").search(name).group(1)
                except:
                    import pdb; pdb.set_trace()
            elif "Yes" in name:
                name = re.compile("Yes, (.*)\.").search(name).group(1)
            elif name.endswith("."):
                name = re.compile("(.*)\.").search(name).group(1)
            else:
                name = name
            names.append(name)
    return names

def read_files(f_dir):
    lines = []
    for fle in tqdm(glob(f_dir)):
        with open(fle, 'r') as fh:
            lines.append((fh.read(), fle))
    return lines

# train_names = get_names(train_dir)
# test_names = get_names(test_dir)


def in_dials(dials, name):
    for dial, fle in dials:
        if name in dial:
            return True
    return False            

# first name
train_dials = read_files(train_dir)
name_in_train = []
for name in tqdm(df.name):
    name = name.split()[0]
    if in_dials(train_dials, name):
        if name not in name_in_train:
            name_in_train.append(name)

name_not_intrain = [n for n in unique_names if n not in name_in_train]
np.random.seed(0)
np.random.shuffle(name_not_intrain)
assert len(name_not_intrain) == len(set(name_not_intrain))
assert len(set(name_not_intrain).intersection(set(name_in_train))) == 0
print(f"first name in train: {len(name_in_train)}")
print(f"first name not in train: {len(name_not_intrain)}")

with open("attacks/membership_inference/candidates/dialog-first-name-US-census/test/test.txt", "w") as fh:
    fh.writelines([" "+ t +"\n" for t in name_not_intrain])

with open("attacks/membership_inference/candidates/dialog-first-name-US-census/train/train.txt", "w") as fh:
    fh.writelines([" "+ t +"\n" for t in name_in_train])