import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def foward(self, x, lengths):
        embedded = self.dropout(self.embed(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max(lengths))

        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedded_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedded_size)
        self.atten = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embedded_size, dec_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)


    def create_mask(self, x_len, y_len):
        max_x_len = x_len.max()
        max_y_len = y_len.max()

        batch_size = len(x_len)

        x_mask = (torch.arange(max_x_len.item())[None, :] < x_len[:, None]).float()
        y_mask = (torch.arange(max_y_len.item())[None, :] < y_len[:, None]).float()

        mask = (1 - y_mask[:, :, None] * x_mask[:, None, :]) != 0

        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        y_embed = self.dropout(self.embed(y))
        y_packed = nn.utils.rnn.pack_padded_sequence(y_embed, y_lengths, batch_first=True, enforce_sorted=False)
        pack_output, hid = self.rnn(y_packed, hid)
        output_seq, _ = nn.utils.rnn.pad_packed_sequence(pack_output, batch_first=True, total_length=max(y_lengths))

        mask = self.create_mask(ctx_lengths, y_lengths)

        output, atten = self.atten(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), dim=-1)

        return output, atten, hid





