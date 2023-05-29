import torch
import numpy as np
from utils import load_file, tokenize2num
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu


def test(args, model, processor):
    model.eval()
    en_sents, cn_sents = load_file(args.data_dir + 'test.txt', tgt_add_bos=False)
    en_sents, _ = tokenize2num(en_sents, cn_sents,
                               processor.en_tokenizer.word2idx,
                               processor.cn_tokenizer.word2idx,
                               sort_reverse=False)
    top_hypos = []
    test_iteration = tqdm(en_sents, desc='test bleu')
    with torch.no_grad():
        for idx, en_sent in enumerate(test_iteration):
            mb_x = torch.from_numpy(np.array(en_sent).reshape(1, -1)).long().to(args.device)
            mb_x_len = torch.from_numpy(np.array([len(en_sent)])).long().to(args.device)
            bos = torch.Tensor([[processor.cn_tokenizer.word2idx['BOS']]]).long().to(args.device)
            completed_hypo = model.beam_search(mb_x, mb_x_len, bos,
                                               processor.cn_tokenizer.word2idx['EOS'],
                                               topk=args.beam_size,
                                               max_length=args.max_beam_search_length)
            top_hypos.append([processor.cn_tokenizer.id2word[id] for id in completed_hypo[0].value])

    bleu_score = corpus_bleu([[ref] for ref in cn_sents], top_hypos)
    print(f'Corpus BLEU: {bleu_score * 100}')
    return bleu_score
