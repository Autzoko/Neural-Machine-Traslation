#import pkuseg
# pkuseg版本维护跟史一样，不用了！
import jieba
import nltk
import random
import torch
import numpy as np
from collections import Counter

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_file(path, tgt_add_bos=True):
    en = []
    cn = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            if tgt_add_bos:
                cn.append(["BOS"] + jieba.lcut(line[1]) + ["EOS"])
            else:
                cn.append(jieba.lcut(line[1]))

    return en, cn

def build_tokenizer(sentences, args):
    word_count = Counter()
    for sen in sentences:
        for word in sen:
            word_count[word] += 1

    ls = word_count.most_common(args.max_vocab_size)
    word2idx = {word : idx + 2 for idx, (word, _) in enumerate(ls)}
    word2idx['UNK'] = args.UNK_IDX
    word2idx['PAD'] = args.PAD_IDX

    id2word = {v : k for k, v in word2idx.items()}
    total_vocab = len(ls) + 2

    return word2idx, id2word, total_vocab


def tokenize2num(en_sentences, cn_sentences, en_word2idx, cn_word2idx, sort_reverse=True):
    length = len(en_sentences)

    out_en_sents = [[en_word2idx.get(word, 1) for word in sen] for sen in en_sentences]
    out_cn_sents = [[cn_word2idx.get(word, 1) for word in sen] for sen in cn_sentences]

    def sort_sents(sents):
        return sorted(range(len(sents)), key=lambda x : len(sents[x]), reverse=True)

    if sort_reverse:
        sorted_index = sort_sents(out_en_sents)
        out_en_sents = [out_en_sents[idx] for idx in sorted_index]
        out_cn_sents = [out_cn_sents[idx] for idx in sorted_index]

    return out_cn_sents, out_cn_sents


def prepare_data(seqs):
    batch_size = len(seqs)
    lengths = [len(seq) for seq in seqs]
    max_length = max(lengths)

    x = np.zeros((batch_size, max_length)).astype("int32")
    for idx in range(batch_size):
        x[idx, :lengths[idx]] = seqs[idx]

    x_lengths = np.array(lengths).astype("int32")
    return x, x_lengths


def getminibatches(n, batch_size, shuffle=True):
    minibatches = np.arange(0, n, batch_size)
    if shuffle:
        np.random.shuffle(minibatches)

    result = []
    for idx in minibatches:
        result.append(np.arange(idx, min(n, idx + batch_size)))
    return result
