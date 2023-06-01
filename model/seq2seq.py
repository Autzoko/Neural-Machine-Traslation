import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        context, hid = self.encoder(x, x_lengths)
        output, atten, hid = self.decoder(context, x_lengths, y, y_lengths, hid)
        return output, atten

    def beam_search(self, x, x_lengths, y, EOS_id, topk=5, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        BOS_id = y[0][0].item()
        hypo = [[BOS_id]]
        hypo_scores = torch.zeros(len(hypo), dtype=torch.float, device=y.device)
        completed_hypo = []
        t = 0
        while len(completed_hypo) < topk and t < max_length:
            t += 1
            hypo_num = len(hypo)
            exp_src_encodings = encoder_out.expand(hypo_num, encoder_out.shape[1], encoder_out.shape[2])
            exp_x_lengths = x_lengths.expand(hypo_num)
            exp_hid = hid.expand(hid.shape[0], hypo_num, hid.shape[2])

            output_t, atten_t, exp_hid = self.decoder(
                exp_src_encodings,
                exp_x_lengths,
                torch.tensor(hypo).long().to(y.device),
                torch.ones(hypo_num).long().to(y.device) * t,
                exp_hid
            )

            live_hypo_num = topk - len(completed_hypo)

            contiuating_hypo_scores = (hypo_scores.unsqueeze(1).expand(hypo_num, output_t.shape[-1])
                                       + output_t[:, -1, :].squeeze(1)).view(-1)
            top_cand_hypo_scores, top_cand_hypo_pos = torch.topk(contiuating_hypo_scores, k=live_hypo_num)

            prev_hypo_ids = top_cand_hypo_pos / (output_t.shape[-1])
            hypo_word_ids = top_cand_hypo_pos % (output_t.shape[-1])

            new_hypo = []
            live_hypo_ids = []
            new_hypo_scores = []

            for prev_hypo_id, hypo_word_id, top_cand_hypo_score in zip(prev_hypo_ids, hypo_word_ids,
                                                                       top_cand_hypo_scores):
                prev_hypo_id = prev_hypo_id.item()
                hypo_word_id = hypo_word_id.item()
                top_cand_hypo_score = top_cand_hypo_score.item()

                new_hypo_sent = hypo[int(prev_hypo_id)] + [hypo_word_id]

                if hypo_word_id == EOS_id:
                    completed_hypo.append(Hypothesis(value=new_hypo_sent[1:-1], score=top_cand_hypo_score))
                else:
                    new_hypo.append(new_hypo_sent)
                    live_hypo_ids.append(prev_hypo_id)
                    new_hypo_scores.append(top_cand_hypo_score)

            if len(completed_hypo) == topk:
                break

            hypo = new_hypo
            hypo_scores = torch.tensor(new_hypo_scores, dtype=torch.float, device=y.device)

        if len(completed_hypo) == 0:
            completed_hypo.append(Hypothesis(value=hypo[0][1:], score=hypo_scores[0].item()))

        completed_hypo.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypo
