# Neural Machine Translation
Using Seq2Seq Architecture to translate sentences
from English to Chinese.

This subject is just a demo for trying implementing NMT task, which does
not pursue high performance.

Just for fun.

## Run
Use command:
```shell
python run.py
```
Besides, these arguments are provided to set training, testing and translating
parameters:
- *--data_dir*: path of data set
- *--batch_size*: batch size
- *--do_train*: train the model and save checkpoints
- *--do_test*: test the model by using checkpoints
- *--do_translate*: do translation task by using checkpoints
- *--learning_rate*: set learning rate
- *--dropout*: set dropout rate
- *--num_epoch*: set training epoch number
- *--max_vocab_size*: set maximum length of sentence
- *--embed_size*: set size of tensor for embedding
- *--enc_hidden_size*: set the size of hidden layer in encoder
- *--dec_hidden_size*: set the size of hidden layer in decoder
- *--warmup_steps*: determine the preset step number of warmup training
- *--GRAD_CLIP*
- *--UNK_IDX*
- *--PAD_IDX*
- *--beam_size*
- *--max_beam_search_length*
