#! /bin/bash

python bin/train_elmo.py \
   --train_prefix='./data/cmu_mosei_dataset_fill.pkl' \
   --vocab_file ./data/vocab-2016-09-10.txt \
   --load_dir ./saved_models/1billion \
   --save_dir ./saved_models/1billion_mosei \
   --lr 0.02 \
   --n_epochs 2 \
   --batch_size 48 


