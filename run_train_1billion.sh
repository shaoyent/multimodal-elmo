#! /bin/bash

python bin/train_elmo.py \
   --train_prefix='./data/1-billion/training-monolingual.tokenized.shuffled/*' \
   --vocab_file ./data/vocab-2016-09-10.txt \
   --save_dir ./saved_models/1billion


