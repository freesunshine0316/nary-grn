import tensorflow as tf
import graph_encoder_utils
import entity_utils
import padding_utils
from tensorflow.python.ops import variable_scope
import numpy as np
import random

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

import metric_utils

from pythonrouge.pythonrouge import Pythonrouge
ROUGE_path = '/u/nalln478/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
data_path = '/u/nalln478/pythonrouge/pythonrouge/RELEASE-1.5.5/data'

def _clip_and_normalize(word_probs, epsilon):
    '''
    word_probs: 1D tensor of [vsize]
    '''
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True) # scale preds so that the class probas of each sample sum to 1

def sentence_rouge(reflex, genlex):
    rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, \
            length=50, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000, favor=True, p=0.5)
    genlex = [[genlex,]]
    reflex = [[[reflex,]]]
    setting_file = rouge.setting(files=False, summary=genlex, reference=reflex)
    result = rouge.eval_rouge(setting_file, recall_only=False, ROUGE_path=ROUGE_path, data_path=data_path)
    return result['ROUGE-L-F']

class ModelGraph(object):
    def __init__(self, word_vocab, char_vocab, Edgelabel_vocab, options=None, mode='train'):
        # the value of 'mode' can be:
        #  'train',
        #  'evaluate'
        self.mode = mode

        # is_training controls whether to use dropout
        is_training = True if mode in ('train', ) else False

        self.options = options
        self.word_vocab = word_vocab

        # encode the input instance
        # encoder.graph_hidden [batch, node_num, vsize]
        # encoder.graph_cell [batch, node_num, vsize]
        with tf.variable_scope('bidir_encoder'):
            self.encoder = graph_encoder_utils.GraphEncoder(
                    word_vocab = word_vocab,
                    edge_label_vocab = Edgelabel_vocab,
                    char_vocab = char_vocab,
                    is_training = is_training, options = options)

            variable_scope.get_variable_scope().reuse_variables()

            self.encoder_rev = graph_encoder_utils.GraphEncoder(
                    word_vocab = word_vocab,
                    edge_label_vocab = Edgelabel_vocab,
                    char_vocab = char_vocab,
                    is_training = is_training, options = options)

        with tf.variable_scope('entity_repre'):
            self.entity = entity_utils.Entity(self.encoder.graph_hiddens)
            self.entity_rev = entity_utils.Entity(self.encoder_rev.graph_hiddens)

            batch_size = tf.shape(self.encoder.graph_hiddens)[0]
            node_num = tf.shape(self.encoder.graph_hiddens)[1]
            dim = tf.shape(self.encoder.graph_hiddens)[2]
            entity_num = tf.shape(self.entity.entity_indices)[1]
            entity_size = tf.shape(self.entity.entity_indices)[2]

            self.encoder_dim = options.neighbor_vector_dim * 2
            # [batch, 3, encoder_dim]
            entity_states = tf.concat(
                    [self.entity.entity_states, self.entity_rev.entity_states], 2)
            # [batch, 3*encoder_dim]
            entity_states = tf.reshape(entity_states, [batch_size, entity_num*dim*2])

        # placeholders
        self.nodes = self.encoder.passage_nodes
        self.nodes_num = self.encoder.passage_nodes_size
        if options.with_char:
            self.nodes_chars = self.encoder.passage_nodes_chars
            self.nodes_chars_num = self.encoder.passage_nodes_chars_size
        self.nodes_mask = self.encoder.passage_nodes_mask

        self.in_neigh_indices = self.encoder.passage_in_neighbor_indices
        self.in_neigh_hidden_indices = self.encoder.passage_in_neighbor_hidden_indices
        self.in_neigh_edges = self.encoder.passage_in_neighbor_edges
        self.in_neigh_mask = self.encoder.passage_in_neighbor_mask

        # rev placeholders
        self.rev_nodes = self.encoder_rev.passage_nodes
        self.rev_nodes_num = self.encoder_rev.passage_nodes_size
        if options.with_char:
            self.rev_nodes_chars = self.encoder_rev.passage_nodes_chars
            self.rev_nodes_chars_num = self.encoder_rev.passage_nodes_chars_size
        self.rev_nodes_mask = self.encoder_rev.passage_nodes_mask

        self.rev_in_neigh_indices = self.encoder_rev.passage_in_neighbor_indices
        self.rev_in_neigh_hidden_indices = self.encoder_rev.passage_in_neighbor_hidden_indices
        self.rev_in_neigh_edges = self.encoder_rev.passage_in_neighbor_edges
        self.rev_in_neigh_mask = self.encoder_rev.passage_in_neighbor_mask


        w_linear = tf.get_variable("w_linear",
                [options.entity_num*self.encoder_dim, options.class_num], dtype=tf.float32)
        b_linear = tf.get_variable("b_linear",
                [options.class_num], dtype=tf.float32)
        # [batch, class_num]
        prediction = tf.nn.softmax(tf.matmul(entity_states, w_linear) + b_linear)
        prediction = _clip_and_normalize(prediction, 1.0e-6)

        ## calculating accuracy
        self.answers = tf.placeholder(tf.int32, [None,])
        self.accu = tf.reduce_sum(
                tf.cast(
                    tf.equal(tf.argmax(prediction,axis=-1,output_type=tf.int32),self.answers),
                    dtype=tf.float32))

        ## calculating loss
        # xent: [batch]
        xent = -tf.reduce_sum(
                tf.one_hot(self.answers,options.class_num)*tf.log(prediction),
                axis=-1)
        self.loss = tf.reduce_mean(xent)

        if mode != 'train':
            print('Return from here, just evaluate')
            return

        if options.optimize_type == 'adadelta':
            clipper = 50
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        elif options.optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    def execute(self, sess, batch, batch_rev, options, is_train=False):
        feed_dict = {}
        # mono
        feed_dict[self.nodes] = batch.nodes
        feed_dict[self.nodes_num] = batch.node_num
        if options.with_char:
            feed_dict[self.nodes_chars] = batch.nodes_chars
            feed_dict[self.nodes_chars_num] = batch.nodes_chars_num

        feed_dict[self.in_neigh_indices] = batch.in_neigh_indices
        feed_dict[self.in_neigh_hidden_indices] = batch.in_neigh_hidden_indices
        feed_dict[self.in_neigh_edges] = batch.in_neigh_edges
        feed_dict[self.in_neigh_mask] = batch.in_neigh_mask

        # rev
        feed_dict[self.rev_nodes] = batch_rev.nodes
        feed_dict[self.rev_nodes_num] = batch_rev.node_num
        if options.with_char:
            feed_dict[self.rev_nodes_chars] = batch_rev.nodes_chars
            feed_dict[self.rev_nodes_chars_num] = batch_rev.nodes_chars_num

        feed_dict[self.rev_in_neigh_indices] = batch_rev.in_neigh_indices
        feed_dict[self.rev_in_neigh_hidden_indices] = batch_rev.in_neigh_hidden_indices
        feed_dict[self.rev_in_neigh_edges] = batch_rev.in_neigh_edges
        feed_dict[self.rev_in_neigh_mask] = batch_rev.in_neigh_mask

        # mono
        feed_dict[self.entity.entity_indices] = batch.entity_indices
        feed_dict[self.entity.entity_indices_mask] = batch.entity_indices_mask

        # rev
        feed_dict[self.entity_rev.entity_indices] = batch_rev.entity_indices
        feed_dict[self.entity_rev.entity_indices_mask] = batch_rev.entity_indices_mask

        feed_dict[self.answers] = batch.y

        if is_train:
            return sess.run([self.accu, self.loss, self.train_op], feed_dict)
        else:
            return sess.run([self.accu, self.loss], feed_dict)


if __name__ == '__main__':
    summary = " Tokyo is the one of the biggest city in the world."
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
    print sentence_rouge(reference, summary)

