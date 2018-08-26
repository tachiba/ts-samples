# Prerequisites

https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

```
DATA=path to simple-examples/data
python ptb_word_lm.py --data_path=$DATA --model=small --num_gpus=0
```