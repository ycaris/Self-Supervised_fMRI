import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, feature_size=116, emb_dim=16, num_layers=2, dropout=0.1, return_states=False):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=emb_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.Linear(emb_dim, 1)
        self.dropout = nn.Dropout(dropout) 
        self.return_states = return_states
        self.init_weights()

    def forward(self, x, device):
        x, _ = self.lstm(x)  # h0, c0 initialized to zero;
        # ignore last hidden state
        y = self.decoder(x)
        y = self.dropout(y)
        y = y.mean(dim=1)

        if self.return_states:
            return x, y
        else:
            return y

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
