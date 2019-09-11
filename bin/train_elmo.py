import os,sys
sys.path.insert(0, os.getcwd())

import pickle

import argparse
import tensorflow as tf

from mbilm.training import train, load_options_latest_checkpoint, load_vocab
from mbilm.data import BidirectionalLMDataset, MultimodalDataset

options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},

     'acou_cnn': {
      'activation': 'relu', 
      'acoustics': {'dim': 74, 
                    'name':['F0','VUV','NAQ','QOQ','H1H2','PSP','MDQ','peakSlope','Rd', 
                        'Rd_conf','creak','MCEP_0','MCEP_1','MCEP_2','MCEP_3','MCEP_4','MCEP_5', 
                        'MCEP_6','MCEP_7','MCEP_8','MCEP_9','MCEP_10','MCEP_11','MCEP_12', 
                        'MCEP_13','MCEP_14','MCEP_15','MCEP_16','MCEP_17','MCEP_18', 
                        'MCEP_19','MCEP_20','MCEP_21','MCEP_22','MCEP_23','MCEP_24',
                        'HMPDM_0','HMPDM_1','HMPDM_2','HMPDM_3','HMPDM_4','HMPDM_5', 
                        'HMPDM_6','HMPDM_7','HMPDM_8','HMPDM_9','HMPDM_10','HMPDM_11','HMPDM_12', 
                        'HMPDM_13','HMPDM_14','HMPDM_15','HMPDM_16','HMPDM_17','HMPDM_18', 
                        'HMPDM_19','HMPDM_20','HMPDM_21','HMPDM_22','HMPDM_23','HMPDM_24',
                        'HMPDD_0','HMPDD_1','HMPDD_2','HMPDD_3','HMPDD_4','HMPDD_5', 
                        'HMPDD_6','HMPDD_7','HMPDD_8','HMPDD_9','HMPDD_10','HMPDD_11','HMPDD_12']},
      'max_acoustic_size_per_token': 50,
      'filters': [[3, 32],
       [3, 32],
       [3, 16] 
       ],
      'n_highway': 0,
      },
    
     'dropout': 0.1,

     'combine':'sigmoid',
     # 'combine':'multiply',
     # 'combine':'concat',
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'learning_rate': 0.2,
     # 'n_train_tokens': 768648884,
     'batch_size': 64,
     'n_tokens_vocab': 793471,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }



def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = args.batch_size  # batch size for each GPU
    n_gpus = args.n_gpus

    prefix = args.train_prefix

    if '1-billion' in prefix :
        train_1b = True
        n_train_tokens = 768648884 # 1-billion

        data = BidirectionalLMDataset(prefix, vocab, test=False,
                                          shuffle_on_load=True)
    else :
        train_1b = False
        n_train_tokens = 651037 # mosei

        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI.cmu_mosei_std_folds import standard_test_fold
        test_split = standard_test_fold
        data = MultimodalDataset(prefix, args.vocab_file, exclude_split=test_split)


    # number of tokens in training data (this for 1B Word Benchmark)

    options['n_train_tokens'] = n_train_tokens
    options['n_tokens_vocab'] = vocab.size
    options['batch_size'] = batch_size

    if args.lr is not None :
        options['learning_rate'] = args.lr
    if args.n_epochs is not None :
        options['n_epochs'] = args.n_epochs


    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, restart_ckpt_file=tf.train.latest_checkpoint(args.load_dir) if args.load_dir is not None else None)


if __name__ == '__main__':     
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files', default='../../saved_models/elmo-tf/checkpoints/multimodal_1')
    parser.add_argument('--load_dir', help='Location of pre-trained checkpoint files', default=None)
    parser.add_argument('--vocab_file', help='Vocabulary file', default='./data/vocab-2016-09-10.txt')
    parser.add_argument('--train_prefix', help='Prefix for train files', default='./data/CMU-MOSEI/CMU-MOSEI-Processed/Transcripts/segmented_text/*')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=48)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('--lr', type=float, help='Learning rate', default=None)
    parser.add_argument('--n_gpus', type=int, help='Number of GPUs to use', default=1)

    args = parser.parse_args()
    main(args)

