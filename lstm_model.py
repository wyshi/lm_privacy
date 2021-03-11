import torch
from torch import nn
from opacus.layers import DPLSTM
import torch.nn.functional as F

class DPLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        vocab_size,
        num_lstm_layers=1,
        bidirectional=False,
        tie_weights=True,
        dp=True,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if True:
            self.lstm = DPLSTM(
                embedding_size,
                hidden_size,
                num_layers=num_lstm_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_lstm_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size != embedding_size:
                raise ValueError(f'When using the tied flag, hidden_size({hidden_size}) must be equal to embedding_size({embedding_size})')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden=None):
        # import pdb
        # pdb.set_trace()
        emb = self.encoder(x)  # -> [B, T, D]
        output, hidden = self.lstm(emb, hidden)  # -> [B, T, H]
        # x = x[:, -1, :]  # -> [B, H]
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

