#! /bin/bash

python bin/extract_embeddings.py \
   --test_prefix='./data/cmu_mosei_dataset_fill.pkl' \
   --vocab_file ./data/vocab-2016-09-10.txt \
   --load_dir ./saved_models/1billion_mosei \
   --out_file='./data/cmu_mosei_melmo.pkl' \
   --batch_size 20 \
   --unroll_steps 376 



