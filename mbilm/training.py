'''
Train and test bidirectional language models.
'''

import os
import time
import json
import re

import tensorflow as tf
import numpy as np

import pickle

# from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from .models import MultimodalModel

from .data import Vocabulary, UnicodeCharsVocabulary, InvalidNumberOfCharacters, BidirectionalLMDataset


tf.logging.set_verbosity(tf.logging.INFO)


def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over 
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))
    
    return average_grads


def summary_gradient_updates(grads, opt, lr):
    '''get summary ops for the magnitude of gradient updates'''

    # strategy:
    # make a dict of variable name -> [variable, grad, adagrad slot]
    vars_grads = {}
    for v in tf.trainable_variables():
        vars_grads[v.name] = [v, None, None]
    for g, v in grads:
        vars_grads[v.name][1] = g
        vars_grads[v.name][2] = opt.get_slot(v, 'accumulator')

    # now make summaries
    ret = []
    for vname, (v, g, a) in vars_grads.items():

        if g is None:
            continue

        if isinstance(g, tf.IndexedSlices):
            # a sparse gradient - only take norm of params that are updated
            values = tf.gather(v, g.indices)
            updates = lr * g.values
            if a is not None:
                updates /= tf.sqrt(tf.gather(a, g.indices))
        else:
            values = v
            updates = lr * g
            if a is not None:
                updates /= tf.sqrt(a)

        values_norm = tf.sqrt(tf.reduce_sum(v * v)) + 1.0e-7
        updates_norm = tf.sqrt(tf.reduce_sum(updates * updates))
        ret.append(
                tf.summary.scalar('UPDATE/' + vname.replace(":", "_"), updates_norm / values_norm))

    return ret

def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)


def _get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional):
    feed_dict = {}
    if not char_inputs:
        token_ids = X['token_ids'][start:end]
        feed_dict[model.token_ids] = token_ids
    else:
        # character inputs
        char_ids = X['tokens_characters'][start:end]
        feed_dict[model.tokens_characters] = char_ids

    if bidirectional:
        if not char_inputs:
            feed_dict[model.token_ids_reverse] = \
                X['token_ids_reverse'][start:end]
        else:
            feed_dict[model.tokens_characters_reverse] = \
                X['tokens_characters_reverse'][start:end]

    if 'tokens_acoustic' in X :
        feed_dict[model.tokens_acoustic] = X['tokens_acoustic'][start:end]
        if bidirectional: 
            feed_dict[model.tokens_acoustic_reverse] = X['tokens_acoustic_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]

    return feed_dict


def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None):

    if not os.path.exists(tf_save_dir):
        os.makedirs(tf_save_dir)

    # not restarting so save the options
    with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
        fout.write(json.dumps(options))

    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', options['learning_rate'])
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = MultimodalModel(options, True, lm_training=isinstance(data, BidirectionalLMDataset))
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    # keep track of loss across all GPUs
                    train_perplexity += loss

        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        input_summaries = [tf.summary.scalar("acoustic_input/input", tf.reduce_mean(model.tokens_acoustic))]
        # input_summaries.append(tf.summary.scalar("acoustic_input/embedding_acoustic", tf.reduce_mean(model.embedding_acoustic)))
        # var1 = [var for var in tf.global_variables() if "lm/MME/CNN_ACO/W_cnn_0:0" in var.name][0]
        # input_summaries.append(tf.summary.scalar("acoustic_input/aco_cnn_weight", tf.reduce_mean(var1)))
        # print(var1.name)
        # var1 = [var for var in tf.global_variables() if "lm/CNN/W_cnn_0:0" in var.name][0]
        # print(var1.name)
        # input_summaries.append(tf.summary.scalar("acoustic_input/lex_cnn_weight", tf.reduce_mean(var1)))

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summmary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries + input_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initialize_all_variables()

    # do the training loop
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            try :
                loader = tf.train.Saver()
                loader.restore(sess, restart_ckpt_file)
                print("Loaded from restart checkpoint")
            except :
                variables_to_restore =  [v for v in tf.global_variables() if 'MME' not in v.name]
                loader = tf.train.Saver(variables_to_restore)
                loader.restore(sess, restart_ckpt_file)
                print("Loaded from restart checkpoint")
            
        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in options
        if char_inputs:
            max_chars = options['char_cnn']['max_characters_per_token']
            print("Using char inputs")

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in models
            }

        feed_dict.update({
            model.tokens_acoustic: np.zeros([batch_size, unroll_steps, options['acou_cnn']['max_acoustic_size_per_token'], options['acou_cnn']['acoustics']['dim']])
        })
        if bidirectional:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                    for model in models
                })
            feed_dict.update({
                model.tokens_acoustic_reverse: np.zeros([batch_size, unroll_steps, options['acou_cnn']['max_acoustic_size_per_token'], options['acou_cnn']['acoustics']['dim']])
            })

        init_state_values, init_embed_acoustic = sess.run([init_state_tensors, model.embedding_acoustic], feed_dict=feed_dict)

        t1 = time.time()
        end_training = False
        for epoch_no in range(options['n_epochs']) :
            data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
            for batch_no, batch in enumerate(data_gen, start=1):
                # slice the input in the batch for the feed_dict
                X = batch
                feed_dict = {}
                for s1, s2 in zip(init_state_tensors, init_state_values) :
                    feed_dict.update({t: v for t, v in zip(s1, s2)})

                for k in range(n_gpus):
                    model = models[k]
                    start = k * batch_size
                    end = (k + 1) * batch_size

                    feed_dict.update(
                        _get_feed_dict_from_X(X, start, end, model,
                                              char_inputs, bidirectional)
                    )

                    if 'tokens_acoustic' not in X:
                        # Add dummy acoustic
                        feed_dict.update({
                            model.tokens_acoustic:
                                np.zero([batch_size, unroll_steps, 50, 74]),
                            model.tokens_acoustic_reverse:
                                np.zero([batch_size, unroll_steps, 50, 74])
                        })

                # This runs the train_op, summaries and the "final_state_tensors"
                #   which just returns the tensors, passing in the initial
                #   state tensors, token ids and next token ids
                if batch_no % 500 != 0:
                    ret = sess.run(
                        [train_op, summary_op, train_perplexity] +
                                                    final_state_tensors,
                        feed_dict=feed_dict
                    )

                    # first three entries of ret are:
                    #  train_op, summary_op, train_perplexity
                    # last entries are the final states -- set them to
                    # init_state_values
                    # for next batch
                    init_state_values = ret[3:]

                else:
                    # also run the histogram summaries
                    ret = sess.run(
                        [train_op, summary_op, train_perplexity, hist_summary_op] + 
                                                    final_state_tensors,
                        feed_dict=feed_dict
                    )
                    init_state_values = ret[4:]
                    

                if batch_no % 500 == 0:
                    summary_writer.add_summary(ret[3], batch_no)
                if batch_no % 100 == 0:
                    # write the summaries to tensorboard and display perplexity
                    summary_writer.add_summary(ret[1], batch_no)
                    print("Epoch %s Batch %s, train_perplexity=%s" % (epoch_no, batch_no, ret[2]))
                    print("Total time: %s" % (time.time() - t1))

                if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                    # save the model
                    print("Saving model")
                    checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

                # embed_acoustic = sess.run(model.embedding_acoustic, feed_dict=feed_dict)

                if batch_no == n_batches_total:
                    # done training!
                    end_training = True


            print("Saving model at end of epoch")
            checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

            if end_training :
                print("End of training")
                break


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip 
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    max_acou = options['acou_cnn']['max_acoustic_size_per_token']
    acou_dim = options['acou_cnn']['acoustics']['dim']

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = unroll_steps
            test_options['dropout'] = 0
            # model = LanguageModel(test_options, False)
            model = MultimodalModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state
        if not char_inputs:
            feed_dict = {
                model.token_ids:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
            }
            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                })  
        else:
            feed_dict = {
                model.tokens_acoustic:
                   np.zeros([batch_size, unroll_steps, max_acou, acou_dim],
                                 dtype=np.float),
                model.tokens_characters:
                   np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
            }
            if bidirectional:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                            dtype=np.int32)
                })

        feed_dict.update({
            model.tokens_acoustic: np.zeros([batch_size, unroll_steps, test_options['acou_cnn']['max_acoustic_size_per_token'], test_options['acou_cnn']['acoustics']['dim']])
        })
        if bidirectional :
            feed_dict.update({
                model.tokens_acoustic_reverse: np.zeros([batch_size, unroll_steps, test_options['acou_cnn']['max_acoustic_size_per_token'], test_options['acou_cnn']['acoustics']['dim']])
            })

        init_state_values = sess.run(
            init_state_tensors,
            feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_batches(batch_size, unroll_steps), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {}
            for s1, s2 in zip(init_state_tensors, init_state_values) :
                feed_dict.update({t: v for t, v in zip(s1, s2)})

            feed_dict.update(
                _get_feed_dict_from_X(X, 0, X['token_ids'].shape[0], model, 
                                          char_inputs, bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            print("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))


    avg_loss = np.mean(batch_losses)
    print("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))

    return np.exp(avg_loss)

def extract(options, ckpt_file, data, batch_size=256, unroll_steps=20, outfile='extracted_dataset.pkl'):
    '''
    Extract embeddings from the model
    '''

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    max_acou = options['acou_cnn']['max_acoustic_size_per_token']
    acou_dim = options['acou_cnn']['acoustics']['dim']

    config = tf.ConfigProto(allow_soft_placement=True)

    new_dataset = {}
    new_dataset['embeddings'] = {}
    new_dataset['labels'] = {}

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = unroll_steps
            test_options['dropout'] = 0
            # model = LanguageModel(test_options, False)
            model = MultimodalModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
            }
            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                })  
        else:
            feed_dict = {
                model.tokens_acoustic:
                   np.zeros([batch_size, unroll_steps, max_acou, acou_dim],
                                 dtype=np.float),
                model.tokens_characters:
                   np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
            }
            if bidirectional:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                            dtype=np.int32)
                })

        feed_dict.update({
            model.tokens_acoustic: np.zeros([batch_size, unroll_steps, test_options['acou_cnn']['max_acoustic_size_per_token'], test_options['acou_cnn']['acoustics']['dim']])
        })
        if bidirectional :
            feed_dict.update({
                model.tokens_acoustic_reverse: np.zeros([batch_size, unroll_steps, test_options['acou_cnn']['max_acoustic_size_per_token'], test_options['acou_cnn']['acoustics']['dim']])
            })

        init_state_values = sess.run(
            init_state_tensors,
            feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_sentences(batch_size, unroll_steps), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {}
            for s1, s2 in zip(init_state_tensors, init_state_values) :
                feed_dict.update({t: v for t, v in zip(s1, s2)})
            # feed_dict = {t: v for t, v in zip(
            #                             init_state_tensors, init_state_values)}

            feed_dict.update(
                _get_feed_dict_from_X(X, 0, X['token_ids'].shape[0], model, 
                                          char_inputs, bidirectional)
            )

            ret = sess.run(
                [final_state_tensors, model.elmo_outputs],
                feed_dict=feed_dict
            )

            init_state_values, elmo_outputs = ret


            for i in range(batch_size):
                sentence_length = batch['lengths'][i]
                if sentence_length == 0:
                    print("Skipping 0 length sentence at {}".format(i))
                    continue
                key = batch['keys'][i]
                new_dataset['embeddings'][key] = [np.average(elmo_outputs[j][i, 0:sentence_length], axis=0) for j in range(len(elmo_outputs))]
                new_dataset['labels'][key] = batch['labels_emotions'][i]


    pickle.dump(new_dataset, open(outfile, 'wb'))
    

    print("Finished extracting embeddings")

    return 



def load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    return options, ckpt_file


def load_vocab(vocab_file, max_word_length=None):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)


def dump_weights(tf_save_dir, outfile):
    '''
    Dump the trained weights from a model to a HDF5 file.
    '''
    import h5py

    def _get_outname(tf_name):
        outname = re.sub(':0$', '', tf_name)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)
        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)
        return outname

    options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.variable_scope('lm'):
            # model = LanguageModel(options, False)
            model = MultimodalModel(options, True)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        with h5py.File(outfile, 'w') as fout:
            for v in tf.trainable_variables():
                if v.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                outname = _get_outname(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, outname))
                shape = v.get_shape().as_list()
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = sess.run([v])[0]
                dset[...] = values

