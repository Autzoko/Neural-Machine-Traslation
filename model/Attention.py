import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(2 * enc_hidden_size, dec_hidden_size)
        self.linear_out = nn.Linear(2 * enc_hidden_size + dec_hidden_size, dec_hidden_size)


    def forward(self, output, context, mask):
        batch_size = context.shape[0]
        enc_seq = context.shape[1]
        dec_seq = output.shape[1]

        context_in = self.linear_in(context.reshape(batch_size * enc_seq, -1).contiguous())
        context_in = context_in.view(batch_size, enc_seq, -1).contiguous()
        atten = torch.bmm(output, context_in.transpose(1, 2))

        atten.data.masked_fill(mask, -1e6)
        atten = F.softmax(atten, dim=2)

        context = torch.bmm(atten, context)
        output = torch.tanh(self.linear_out(output.view(batch_size * dec_seq, -1))).view(batch_size, dec_seq, -1)

        return output, atten
