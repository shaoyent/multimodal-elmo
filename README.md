# Multimodal Embeddings from Language Models

This repo implements the models described in https://arxiv.org/abs/1909.04302

## Requirements

Python 3.7 requirements
```
h5py
numpy
scikit-learn
scipy
sklearn
tqdm
validators
requests
tensorflow-gpu==1.13.1
torch==1.0.1
```

The CMU-Multimodal SDK is also required \
https://github.com/A2Zadeh/CMU-MultimodalSDK

## M-ELMo training

As in ELMo, the lexical portion of the model is trained using the [1 Billion Word Benchmark](http://www.statmt.org/lm-benchmark/). 
After downloading the data the language model can be trained using `run_train_1billion.sh` or the command:
```
python bin/train_elmo.py \
	--train_prefix='./data/1-billion/training-monolingual.tokenized.shuffled/*' \
	--vocab_file ./data/vocab-2016-09-10.txt \
	--save_dir ./saved_models/1billion
```

The CMU-MOSEI dataset is aligned and processed using the CMU-MultimodalSDK. To prepare the data and train the multimodal biLM run :
```
python data/prepare_mosei.py

python bin/train_elmo.py \
   --train_prefix='./data/cmu_mosei_dataset_fill.pkl' \
   --vocab_file ./data/vocab-2016-09-10.txt \
   --load_dir ./saved_models/1billion \
   --save_dir ./saved_models/1billion_mosei \
   --lr 0.02 \
   --n_epochs 2 \
   --batch_size 48
```

## Evaluating on CMU-MOSEI

To extract multimodal embeddings from CMU-MOSEI run the script `run_extract_mosei.sh` or use the command:
```
python bin/extract_embeddings.py \
   --test_prefix='./data/cmu_mosei_dataset_fill.pkl' \
   --vocab_file ./data/vocab-2016-09-10.txt \
   --load_dir ./saved_models/1billion_mosei \
   --out_file='./data/cmu_mosei_melmo.pkl' \
   --batch_size 20 \
   --unroll_steps 376
```

## Pretrained models

## Credits

The base ELMo structure was forked from https://github.com/allenai/bilm-tf



