import os,sys
sys.path.insert(0, os.getcwd())

import argparse

from mbilm.training import extract, load_options_latest_checkpoint, load_vocab
from mbilm.data import MultimodalDataset

def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.load_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.vocab_file, max_word_length)

    test_prefix = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    # if options.get('bidirectional'):
    #     data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    # else:
    #     data = LMDataset(test_prefix, vocab, **kwargs)

    data = MultimodalDataset(test_prefix, args.vocab_file)

    extract(options, ckpt_file, data, batch_size=args.batch_size, unroll_steps=args.unroll_steps, outfile=args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--load_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--out_file', help='Output file to save embeddings')
    parser.add_argument('--unroll_steps', type=int, default=20)
    parser.add_argument('--batch_size',
        type=int, default=1,
        help='Batch size')

    args = parser.parse_args()
    main(args)

