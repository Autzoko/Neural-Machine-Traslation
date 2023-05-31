import os

import torch

from utils import *


class Tokenizer(object):
    def __init__(self, word2idx, id2word, vocab_size):
        self.word2idx = word2idx
        self.id2word = id2word
        self.vocab_size = vocab_size


class DataProcess(object):
    def __init__(self, args):
        cached_en_tokenizer = os.path.join(args.data_dir, "cached_{}".format("en_tokenizer"))
        cached_cn_tokenizer = os.path.join(args.data_dir, "cached_{}".format("cn_tokenizer"))

        if not os.path.exists(cached_en_tokenizer) or not os.path.exists(cached_cn_tokenizer):
            en_sents, cn_sents = load_file(args.data_dir + "train.txt")
            en_word2idx, en_id2word, en_vocab_size = build_tokenizer(en_sents, args)
            cn_word2idx, cn_id2word, cn_vocab_size = build_tokenizer(cn_sents, args)

            torch.save([en_word2idx, en_id2word, en_vocab_size], cached_en_tokenizer)
            torch.save([cn_word2idx, cn_id2word, cn_vocab_size], cached_cn_tokenizer)
        else:
            en_word2idx, en_id2word, en_vocab_size = torch.load(cached_en_tokenizer)
            cn_word2idx, cn_id2word, cn_vocab_size = torch.load(cached_cn_tokenizer)

        self.en_tokenizer = Tokenizer(en_word2idx, en_id2word, en_vocab_size)
        self.cn_tokenizer = Tokenizer(cn_word2idx, cn_id2word, cn_vocab_size)

    def get_train_examples(self, args):
        return self._create_examples(os.path.join(args.data_dir, "train.txt"), "train", args)

    def get_dev_examples(self, args):
        return self._create_examples(os.path.join(args.data_dir, "dev.txt"), "dev", args)

    def _create_examples(self, path, set_type, args):
        en_sents, cn_sents = load_file(path)
        out_en_sents, out_cn_sents = tokenize2num(en_sents, cn_sents, self.en_tokenizer.word2idx,
                                                  self.cn_tokenizer.word2idx)
        minibatches = getminibatches(len(out_en_sents), args.batch_size)

        all_examples = []
        for minibatch in minibatches:
            mb_en_sentences = [out_en_sents[i] for i in minibatch]
            mb_cn_sentences = [out_cn_sents[i] for i in minibatch]

            mb_x, mb_x_len = prepare_data(mb_en_sentences)
            mb_y, mb_y_len = prepare_data(mb_cn_sentences)

            all_examples.append((mb_x, mb_x_len, mb_y, mb_y_len))
        return all_examples
