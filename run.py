import argparse
import os.path
import torch
import nltk
import train
import test
from utils import *
from utils import DataProcessor
from model import encoder_decoder
from model import seq2seq
from loss import criterion


def run():
    parse = argparse.ArgumentParser()

    parse.add_argument("--data_dir", default='./data/', type=str, required=False)
    parse.add_argument("--batch_size", default=16, type=int)
    parse.add_argument("--do_train", default=True, action="store_true")
    parse.add_argument("--do_test", default=True, action="store_true")
    parse.add_argument("do_translate", default=True, action="store_true")
    parse.add_argument("learning_rate", default=5e-4, type=float)
    parse.add_argument("--dropout", default=0.3, type=float)
    parse.add_argument("--num_epoch", default=10, type=int)
    parse.add_argument("--max_vocab_size", default=50000, type=int)
    parse.add_argument("--embed_size", default=300, type=int)
    parse.add_argument("--enc_hidden_size", default=512, type=int)
    parse.add_argument("--dec_hidden_size", default=512, type=int)
    parse.add_argument("--warmup_steps", default=0, type=int)
    parse.add_argument("--GRAD_CLIP", default=1, type=int)
    parse.add_argument("--UNK_IDX", default=0, type=int)
    parse.add_argument("--PAD_IDX", default=0, type=int)
    parse.add_argument("--beam_size", default=5, type=int)
    parse.add_argument("--max_beam_search_length", default=100, type=int)

    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    setseed(2023)

    processor = DataProcessor.DataProcess(args)
    encoder = encoder_decoder.Encoder(processor.en_tokenizer.vocab_size,
                                      args.embed_size,
                                      args.enc_hidden_size,
                                      args.dec_hidden_size,
                                      args.dropout)
    decoder = encoder_decoder.Decoder(processor.cn_tokenizer.vocab_size,
                                      args.embed_size,
                                      args.enc_hidden_size,
                                      args.dec_hidden_size,
                                      args.dropout)
    model = seq2seq.seq2seq(encoder, decoder)

    if os.path.exists("translate-best.th"):
        model.load_state_dict(torch.load("translate-best.th"))
    model.to(device)

    loss_fn = criterion.ModelCriterion().to(device)

    train_data = processor.get_train_examples(args)
    eval_data = processor.get_dev_examples(args)

    if args.do_train:
        train.train(args, model, train_data, loss_fn, eval_data)

    if args.do_test:
        test.test(args, model, processor)

    if args.do_translate:
        model.load_state_dict(torch.load("translate-best.th"))
        model.to(device)
        while True:
            title = input("请输入要翻译的英文句子：\n")
            if len(title.strip()) == 0:
                continue
            title = ['BOS'] + nltk.word_tokenize(title.lower()) + ['EOS']
            title_num = [processor.en_tokenizer.word2idx.get(word, 1) for word in title]
            mb_x = torch.from_numpy(np.array(title_num).reshape(1, -1)).long().to(device)
            mb_x_len = torch.from_numpy(np.array([len(title_num)])).long().to(device)
            bos = torch.Tensor([[processor.cn_tokenizer.word2idx['BOS']]]).long().to(device)
            completed_hypo = model.beam_search(mb_x, mb_x_len, bos, processor.cn_tokenizer.word2idx['EOS'],
                                               topk=args.beam_size, max_length=args.max_beam_search_length)
            for hypo in completed_hypo:
                result = "".join([processor.cn_tokenizer.id2word[id] for id in hypo.value])
                score = hypo.score
                print(f'翻译结果为：{result};\nScore: {score}')


if __name__ == '__main__':
    run()
