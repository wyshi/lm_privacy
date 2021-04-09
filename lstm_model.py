import torch
from torch import nn
from opacus.layers import DPLSTM
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class DPLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        vocab_size,
        pad_token_id,
        num_lstm_layers=1,
        dropout=0.5,
        bidirectional=False,
        tie_weights=True,
        dp=True,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.nlayers = num_lstm_layers
        self.pad_token_id = pad_token_id

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if True:
            self.lstm = DPLSTM(
                embedding_size,
                hidden_size,
                num_layers=num_lstm_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_lstm_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout
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

    def forward(self, x, seq_lens=None, hidden=None):
        # x = x.transpose(0, 1) # because batch_first is True
        # import pdb; pdb.set_trace()
        emb = self.drop(self.encoder(x))  # -> [B, T, D]
        if seq_lens is not None:
            emb = pack_padded_sequence(emb, seq_lens, batch_first=True, enforce_sorted=False, )
        output, hidden = self.lstm(emb, hidden)  # -> [B, T, H]
        if seq_lens is not None:
            output, output_lens = pad_packed_sequence(output, batch_first=True, padding_value=self.pad_token_id)

        # output = output.transpose(0, 1)
        # x = x[:, -1, :]  # -> [B, H]
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.hidden_size),
                weight.new_zeros(self.nlayers, bsz, self.hidden_size))

