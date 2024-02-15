import torch.nn as nn
import torch
import math


class SimpleTransformer(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size=116, emb_dim=512, num_layers=6, dropout=0.1, nhead=4):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Linear(feature_size, emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(
            max_position=154+1, emb_dim=emb_dim, dropout=dropout)  # longest time point
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.emb_dim = emb_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        mask[:, 0] = 0.0  # make sure the classification token won't be masked out
        return mask

    def forward(self, src, device):

        batch_size, seq_len, _ = src.size()

        # get classication token
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # position enconding of x
        src = self.embedding(src) * math.sqrt(self.emb_dim)
        src = torch.cat((cls_token, src), dim=1)
        src = self.pos_encoder(src, device)

        mask = self._generate_square_subsequent_mask(seq_len).to(device)
        output = self.transformer_encoder(src)

        # # global pooling for transformer
        # output = output.mean(dim=1)
        # output = self.decoder(output)

        # classification token for output
        output = output[:, 0]
        output = self.decoder(output)

        return output


# Positional Encoding to inject some information about the relative or absolute position of the tokens in the sequence.
# Use absolute encoding here

class PositionalEncoding(nn.Module):
    def __init__(self, max_position=154, emb_dim=1024, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(
            self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(
            self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x, device):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        x_pos = self.positional_encoding[:batch_size, :seq_len, :].to(device)

        return self.dropout(x + x_pos)
