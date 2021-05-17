import os

paths=[
    "attacks/membership_inference/candidates/dialog-pick-name-nodplr2bs4-without-ground-truth-help",
    "attacks/membership_inference/candidates/dialog-pick-names-dp-without-ground-truth-help",
    "attacks/membership_inference/candidates/dialog-pick-names-without-ground-truth-help"
    ]

test_names = []
train_names = []
for p in paths:
    with open(os.path.join(p, "train", "train.txt"), "r") as fh:
        a = fh.readlines()
        train_names.append(a)
    with open(os.path.join(p, "test", "test.txt"), "r") as fh:
        a = fh.readlines()
        test_names.append(a)

inter_train_names = list(set(train_names[0]).intersection(train_names[1]).intersection(train_names[2]))
inter_test_names = list(set(test_names[0]).intersection(test_names[1]).intersection(test_names[2]))


print(f"first name in train: {len(inter_train_names)}")
print(f"first name not in train: {len(inter_test_names)}")

with open("attacks/membership_inference/candidates/dialog-pick-first-name-intersection/test/test.txt", "w") as fh:
    fh.writelines([t for t in inter_test_names])

with open("attacks/membership_inference/candidates/dialog-pick-first-name-intersection/train/train.txt", "w") as fh:
    fh.writelines([t for t in inter_train_names])