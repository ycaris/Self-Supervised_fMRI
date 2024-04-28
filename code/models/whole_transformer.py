import torch.nn as nn
import torch
import math


class WholeTransformer(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size=116, emb_dim=1024, num_layers=6, dropout=0.1, nhead=4, time_period=48):
        super(WholeTransformer, self).__init__()

        self.embedding = nn.Linear(feature_size, emb_dim)
        self.emb_dim = emb_dim
        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        # transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=2)

        # other stuff
        self.pos_encoder = PositionalEncoding(
            max_position=154, emb_dim=emb_dim, dropout=dropout)  # longest time point
        self.mask = self._generate_square_subsequent_mask(time_period)

        self.decoder = nn.Linear(emb_dim, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, device):

        batch_size, seq_len, _ = src.size()

        # position enconding of x
        src = self.embedding(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src, device)

        tgt = self.embedding(tgt) * math.sqrt(self.emb_dim)
        tgt = self.pos_encoder(tgt, device)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(
            tgt, memory, tgt_mask=self.mask.to(device))
        output = self.decoder(output)

        return output


class WholeTransformerClassification(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size=116, emb_dim=512, num_layers=6, dropout=0.1, nhead=4):
        super(WholeTransformerClassification, self).__init__()

        self.embedding = nn.Linear(feature_size, emb_dim)
        self.emb_dim = emb_dim
        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.pos_encoder = PositionalEncoding(
            max_position=154, emb_dim=emb_dim, dropout=dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, device):

        batch_size, seq_len, _ = src.size()

        # position enconding of x
        src = self.embedding(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src, device)

        output = self.transformer_encoder(src)

        # global pooling for transformer
        output = self.decoder(output)
        output = output.mean(dim=1)

        return output

    def load_from(self, state_dict):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'decoder' in name:
                print("skip decoder")
                continue
            if name in own_state and param.size() == own_state[name].size():
                # If the parameter exists in the pretrained model and sizes match, use the pretrained weight
                param = param.data
                own_state[name].copy_(param)


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
