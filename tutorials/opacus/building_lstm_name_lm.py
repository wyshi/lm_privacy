"""
python /home/wyshi/privacy/tutorials/building_lstm_name_classifier.py  2>&1 | tee tutorials/data/model/log_torch_lstm
"""
import os
import requests

NAMES_DATASET_URL = "https://download.pytorch.org/tutorial/data.zip"
DATA_DIR = "names"

import zipfile
import urllib
import torch.nn.functional as F

PAD_IDX = 258

# dataset
def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting ...")
    filename = "data.zip"

    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)
    print("Completed!")

download_and_extract(NAMES_DATASET_URL, DATA_DIR)


names_folder = os.path.join(DATA_DIR, 'data', 'names')
all_filenames = []

for language_file in os.listdir(names_folder):
    all_filenames.append(os.path.join(names_folder, language_file))
    
print(os.listdir(names_folder))


# model 
import torch
import torch.nn as nn

class CharByteEncoder(nn.Module):
    """
    This encoder takes a UTF-8 string and encodes its bytes into a Tensor. It can also
    perform the opposite operation to check a result.
    Examples:
    >>> encoder = CharByteEncoder()
    >>> t = encoder('Ślusàrski')  # returns tensor([256, 197, 154, 108, 117, 115, 195, 160, 114, 115, 107, 105, 257])
    >>> encoder.decode(t)  # returns "<s>Ślusàrski</s>"
    """

    def __init__(self):
        super().__init__()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        self.start_idx = 256
        self.end_idx = 257
        self.pad_idx = PAD_IDX

    def forward(self, s: str, pad_to=0) -> torch.LongTensor:
        """
        Encodes a string. It will append a start token <s> (id=self.start_idx) and an end token </s>
        (id=self.end_idx).
        Args:
            s: The string to encode.
            pad_to: If not zero, pad by appending self.pad_idx until string is of length `pad_to`.
                Defaults to 0.
        Returns:
            The encoded LongTensor of indices.
        """
        encoded = s.encode()
        n_pad = pad_to - len(encoded) if pad_to > len(encoded) else 0
        return torch.LongTensor(
            [self.start_idx]
            + [c for c in encoded]  # noqa
            + [self.end_idx]
            + [self.pad_idx for _ in range(n_pad)]
        )

    def decode(self, char_ids_tensor: torch.LongTensor) -> str:
        """
        The inverse of `forward`. Keeps the start, end and pad indices.
        """
        char_ids = char_ids_tensor.cpu().detach().tolist()

        out = []
        buf = []
        for c in char_ids:
            if c < 256:
                buf.append(c)
            else:
                if buf:
                    out.append(bytes(buf).decode())
                    buf = []
                if c == self.start_idx:
                    out.append(self.start_token)
                elif c == self.end_idx:
                    out.append(self.end_token)
                elif c == self.pad_idx:
                    out.append(self.pad_token)

        if buf:  # in case some are left
            out.append(bytes(buf).decode())
        return "".join(out)

    def __len__(self):
        """
        The length of our encoder space. This is fixed to 256 (one byte) + 3 special chars
        (start, end, pad).
        Returns:
            259
        """
        return 259


# train/valication preparation

from torch.nn.utils.rnn import pad_sequence

def padded_collate(batch, padding_idx=PAD_IDX):
    x = pad_sequence(
        [elem[0] for elem in batch], batch_first=True, padding_value=padding_idx
    )
    y = pad_sequence(
        [elem[1] for elem in batch], batch_first=True, padding_value=padding_idx
    )

    return x, y




from torch.utils.data import Dataset
from pathlib import Path


class NamesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.labels = list({langfile.stem for langfile in self.root.iterdir()})
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.encoder = CharByteEncoder()
        self.samples = self.construct_samples()

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def construct_samples(self):
        samples = []
        for langfile in self.root.iterdir():
            label_name = langfile.stem
            label_id = self.labels_dict[label_name]
            with open(langfile, "r") as fin:
                for row in fin:
                    s = self.encoder(row.strip())
                    samples.append(
                        (s[:-1], s[1:])
                    )
        return samples

    def label_count(self):
        cnt = Counter()
        for _x, y in self.samples:
            label = self.labels[int(y)]
            cnt[label] += 1
        return cnt


VOCAB_SIZE = 256 + 3  # 256 alternatives in one byte, plus 3 special characters.




secure_rng = False
train_split = 0.8
test_every = 5
batch_size = 800

ds = NamesDataset(names_folder)
train_len = int(train_split * len(ds))
test_len = len(ds) - train_len

print(f"{train_len} samples for training, {test_len} for testing")

if secure_rng:
    try:
        import torchcsprng as prng
    except ImportError as e:
        msg = (
            "To use secure RNG, you must install the torchcsprng package! "
            "Check out the instructions here: https://github.com/pytorch/csprng#installation"
        )
        raise ImportError(msg) from e

    generator = prng.create_random_device_generator("/dev/urandom")

else:
    generator = None

train_ds, test_ds = torch.utils.data.random_split(
    ds, [train_len, test_len], generator=generator
)




from torch.utils.data import DataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

sample_rate = batch_size / len(train_ds)

train_loader = DataLoader(
    train_ds,
    num_workers=8,
    pin_memory=True,
    generator=generator,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(train_ds),
        sample_rate=sample_rate,
        generator=generator,
    ),
    collate_fn=padded_collate,
)

test_loader = DataLoader(
    test_ds,
    batch_size=2 * batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    collate_fn=padded_collate,
)


# import pdb
# pdb.set_trace()


from statistics import mean
import math

def train(model, criterion, optimizer, train_loader, epoch, device="cuda:0"):
    # accs = []
    losses = []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.view(-1))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        import pdb
        pdb.set_trace()

        # preds = logits.argmax(-1)
        # n_correct = float(preds.eq(y).sum())
        # batch_accuracy = n_correct / len(y)

        # accs.append(batch_accuracy)
        losses.append(float(loss))

    printstr = (
        f"\t Epoch {epoch}. | Loss: {mean(losses):.6f} | ppl: {math.exp(mean(losses)):.6f}"
    )
    try:
        privacy_engine = optimizer.privacy_engine
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        printstr += f" | (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    except AttributeError:
        pass
    print(printstr)
    return


def test(model, test_loader, privacy_engine, device="cuda:0"):
    losses = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y.view(-1))
            losses.append(float(loss))
    printstr = "\n----------------------------\n" f"Test loss: {mean(losses):.6f} | Test ppl: {math.exp(mean(losses)):.6f}"
    if privacy_engine:
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        printstr += f" (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    print(printstr + "\n----------------------------\n")
    return



# Training hyper-parameters
epochs = 50
learning_rate = 2.0

# Privacy engine hyper-parameters
sigma = 1.0
max_per_sample_grad_norm = 1.5
delta = 8e-5





import torch
from torch import nn
from opacus.layers import DPLSTM

class CharNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        output_size,
        num_lstm_layers=1,
        bidirectional=False,
        vocab_size=VOCAB_SIZE,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = DPLSTM(
            embedding_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # -> [B, T, D]
        x, _ = self.lstm(x, hidden)  # -> [B, T, H]
        decoded = self.decoder(x)  # -> [B, T, V]
        decoded = decoded.view(-1, self.vocab_size)  # -> [B, C]
        return F.log_softmax(decoded, dim=1)




# Set the device to run on a GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define classifier parameters
embedding_size = 64
hidden_size = 128  # Number of neurons in hidden layer after LSTM
n_lstm_layers = 1
bidirectional_lstm = False

model = CharNNClassifier(
    embedding_size,
    hidden_size,
    len(ds.labels),
    n_lstm_layers,
    bidirectional_lstm,
).to(device)



criterion = nn.NLLLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)






from opacus import PrivacyEngine

privacy_engine = PrivacyEngine(
    model,
    sample_rate=sample_rate,
    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=sigma,
    max_grad_norm=max_per_sample_grad_norm,
    target_delta=delta,
    secure_rng=secure_rng,
)
import pdb; pdb.set_trace()
privacy_engine.attach(optimizer)
privacy_engine.detach()



from tqdm import tqdm

print("Train stats: \n")
for epoch in tqdm(range(epochs)):
    train(model, criterion, optimizer, train_loader, epoch, device=device)
    if test_every:
        if epoch % test_every == 0:
            test(model, test_loader, privacy_engine, device=device)

test(model, test_loader, privacy_engine, device=device)