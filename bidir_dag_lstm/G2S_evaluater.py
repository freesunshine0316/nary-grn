# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import re
import os
import sys
import json
import time
import numpy as np

import tensorflow as tf
import namespace_utils

import G2S_trainer
import G2S_data_stream
from G2S_model_graph import ModelGraph

from vocab_utils import Vocab


tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the test file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = G2S_trainer.enrich_options(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    edgelabel_vocab = Vocab(model_prefix + ".edgelabel_vocab", fileformat='txt2')
    print('edgelabel_vocab: {}'.format(edgelabel_vocab.word_vecs.shape))
    char_vocab = None
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))

    print('Loading test set from {}.'.format(in_path))
    if FLAGS.infile_format == 'fof':
        testset = G2S_data_stream.read_nary_from_fof(in_path, FLAGS, is_rev=False)
        testset_rev = G2S_data_stream.read_nary_from_fof(in_path, FLAGS, is_rev=True)
    else:
        testset = G2S_data_stream.read_nary_file(in_path, FLAGS, is_rev=False)
        testset_rev = G2S_data_stream.read_nary_file(in_path, FLAGS, is_rev=True)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    batch_size=-1
    devDataStream = G2S_data_stream.G2SDataStream(testset, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=False, batch_size=batch_size)
    devDataStreamRev = G2S_data_stream.G2SDataStream(testset_rev, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=False, batch_size=batch_size)
    assert devDataStream.num_instances == devDataStreamRev.num_instances
    assert devDataStream.num_batch == devDataStreamRev.num_batch
    print('Number of instances in testDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(devDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, Edgelabel_vocab=edgelabel_vocab,
                                         options=FLAGS, mode="evaluate")

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        devDataStream.reset()
        instances = []
        instances_rev = []
        outputs = []
        test_loss = 0.0
        test_right = 0.0
        test_total = 0.0
        start_time = time.time()
        for batch_index in xrange(devDataStream.get_num_batch()): # for each batch
            cur_batch = devDataStream.get_batch(batch_index)
            cur_batch_rev = devDataStreamRev.get_batch(batch_index)
            assert cur_batch.batch_size == cur_batch_rev.batch_size
            assert np.array_equal(cur_batch.node_num, cur_batch_rev.node_num)
            assert np.array_equal(cur_batch.y, cur_batch_rev.y)
            accu_value, loss_value, output_value = valid_graph.execute(sess, cur_batch, cur_batch_rev, FLAGS, is_train=False)
            instances += cur_batch.instances
            instances_rev += cur_batch_rev.instances
            outputs += output_value.flatten().tolist()
            test_loss += loss_value
            test_right += accu_value
            test_total += cur_batch.batch_size
        duration = time.time() - start_time
        print('Decoding time %.3f sec' % (duration))

        assert len(instances) == len(instances_rev) and len(instances) == len(outputs)
        json.dump((instances,instances_rev,outputs,testset,testset_rev), open(out_path,'w'))

        print('Test accu {}, right {}, total {}'.format(1.0*test_right/test_total, test_right, test_total))

