'''
script to find test dialogs (data/simdial/test) that is not in training (data/simdial/train)
python scripts/find_test_dialog.py
'''
from glob import glob
import re
import pandas as pd

test_dir = 'data/simdial/test/*'
train_dir = 'data/simdial/train/*'

df = pd.read_csv('simdial_privacy/database/database_20000.csv')


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
    for fle in glob(f_dir):
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

train_dials = read_files(train_dir)
name_in_train = []
name_not_intrain = []
for name in df.name:
    if in_dials(train_dials, name):
        name_in_train.append(name)
        continue
    name_not_intrain.append(name)

name_not_intrain = list(set(name_not_intrain))
name_in_train = list(set(name_in_train))


print(f"in train: {len(name_in_train)}")
print(f"not in train: {len(name_not_intrain)}")

assert (len(name_in_train) + len(name_not_intrain)) == len(set(df.name))

with open("attacks/membership_inference/candidates/dialog/test/test.txt", "w") as fh:
    fh.writelines([t+"\n" for t in name_not_intrain])

with open("attacks/membership_inference/candidates/dialog/train/train.txt", "w") as fh:
    fh.writelines([t+"\n" for t in name_in_train])