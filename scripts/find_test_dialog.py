'''
script to find test dialogs (data/simdial/test) that is not in training (data/simdial/train)
python scripts/find_test_dialog.py
'''
from glob import glob
import re

test_dir = 'data/simdial/test/*'
train_dir = 'data/simdial/train/*'


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


train_names = get_names(train_dir)
test_names = get_names(test_dir)

a = list(set(test_names) - set(train_names))

with open("attacks/membership_inference/test_dialogs.txt", "w") as fh:
    fh.writelines([s+"\n" for s in a])