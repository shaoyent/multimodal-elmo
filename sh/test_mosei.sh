#! /bin/bash

# F1 30_20_0.4_1000.500_relu_0.001_1.6_mosei_melmo_fill_clean_cnn3_sigmoid_new.pkl_2.8
# WA 30_32_0.5_750.500_relu_0.002_1.1_mosei_melmo_fill_clean_cnn3_sigmoid_new.pkl_3.2
SEEDS="5566 5478 5678 5487 8765 7788 8877 6969 9696 0000"


python eval/mosei/train_mosei.py \
	--seed 5678 \
  	--save_dir ./saved_models/ \
  	--dataset ./data/cmu_mosei_melmo.pkl \
  	--min_ir 2.8 \
  	--gamma 1.6  \
  	--num_epochs 30 \
  	--batch_size 20 \
  	--lr 0.001 \
  	--dropout 0.4 \
  	--layers 1000.500 \
  	--activation relu \
