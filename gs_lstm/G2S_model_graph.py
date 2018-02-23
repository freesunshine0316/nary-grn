import tensorflow as tf
import graph_encoder_utils
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

def collect_by_indices(memory, indices): # [batch, node_num, dim], [batch, 3, 5]
    batch_size = tf.shape(indices)[0]
    entity_num = tf.shape(indices)[1]
    entity_size = tf.shape(indices)[2]
    idxs = tf.range(0, limit=batch_size) # [batch]
    idxs = tf.reshape(idxs, [-1, 1, 1]) # [batch, 1, 1]
    idxs = tf.tile(idxs, [1, entity_num, entity_size])
    indices = tf.maximum(indices, tf.zeros_like(indices, dtype=tf.int32))
    indices = tf.stack((idxs,indices), axis=3) # [batch,3,5,2]
    return tf.gather_nd(memory, indices)

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
        self.encoder = graph_encoder_utils.GraphEncoder(
                word_vocab = word_vocab,
                edge_label_vocab = Edgelabel_vocab,
                char_vocab = char_vocab,
                is_training = is_training, options = options)

        # ============== Choices of attention memory ================
        if options.attention_type == 'hidden':
            self.encoder_dim = options.neighbor_vector_dim
            self.encoder_states = self.encoder.graph_hiddens
        elif options.attention_type == 'hidden_cell':
            self.encoder_dim = options.neighbor_vector_dim * 2
            self.encoder_states = tf.concat([self.encoder.graph_hiddens, self.encoder.graph_cells], 2)
        elif options.attention_type == 'hidden_embed':
            self.encoder_dim = options.neighbor_vector_dim + options.node_dim
            self.encoder_states = tf.concat([self.encoder.graph_hiddens, self.encoder.node_representations], 2)
        else:
            assert False, '%s not supported yet' % options.attention_type

        self.nodes = self.encoder.passage_nodes
        self.nodes_num = self.encoder.passage_nodes_size
        if options.with_char:
            self.nodes_chars = self.encoder.passage_nodes_chars
            self.nodes_chars_num = self.encoder.passage_nodes_chars_size
        self.nodes_mask = self.encoder.passage_nodes_mask

        self.in_neigh_indices = self.encoder.passage_in_neighbor_indices
        self.in_neigh_edges = self.encoder.passage_in_neighbor_edges
        self.in_neigh_mask = self.encoder.passage_in_neighbor_mask

        self.out_neigh_indices = self.encoder.passage_out_neighbor_indices
        self.out_neigh_edges = self.encoder.passage_out_neighbor_edges
        self.out_neigh_mask = self.encoder.passage_out_neighbor_mask

        ## generating prediction results
        self.entity_indices = tf.placeholder(tf.int32, [None, None, None],
                name="entity_indices")
        self.entity_indices_mask = tf.placeholder(tf.float32, [None, None, None],
                name="entity_indices_mask")
        batch_size = tf.shape(self.encoder_states)[0]
        node_num = tf.shape(self.encoder_states)[1]
        dim = tf.shape(self.encoder_states)[2]
        entity_num = tf.shape(self.entity_indices)[1]
        entity_size = tf.shape(self.entity_indices)[2]

        # self.encoder_states [batch, node_num, encoder_dim]
        # entity_states [batch, 3, 5, dim]
        entity_states = collect_by_indices(self.encoder_states, self.entity_indices)
        # applying mask
        entity_states = entity_states * tf.expand_dims(self.entity_indices_mask, axis=-1)
        # average within each entity: [batch, 3, encoder_dim]
        entity_states = tf.reduce_mean(entity_states, axis=2)
        # flatten: [batch, 3*encoder_dim]
        entity_states = tf.reshape(entity_states, [batch_size, entity_num*dim])

        w_linear = tf.get_variable("w_linear",
                [options.entity_num*self.encoder_dim, options.class_num], dtype=tf.float32)
        b_linear = tf.get_variable("b_linear",
                [options.class_num], dtype=tf.float32)
        # [batch, class_num]
        prediction = tf.nn.softmax(tf.matmul(entity_states, w_linear) + b_linear)
        prediction = _clip_and_normalize(prediction, 1.0e-6)
        self.output = tf.argmax(prediction,axis=-1,output_type=tf.int32)

        ## calculating accuracy
        self.refs = tf.placeholder(tf.int32, [None,])
        self.accu = tf.reduce_sum(tf.cast(tf.equal(self.output,self.refs),dtype=tf.float32))

        ## calculating loss
        # xent: [batch]
        xent = -tf.reduce_sum(
                tf.one_hot(self.refs,options.class_num)*tf.log(prediction),
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


    def execute(self, sess, batch, options, is_train=False):
        feed_dict = {}
        feed_dict[self.nodes] = batch.nodes
        feed_dict[self.nodes_num] = batch.node_num
        if options.with_char:
            feed_dict[self.nodes_chars] = batch.nodes_chars
            feed_dict[self.nodes_chars_num] = batch.nodes_chars_num

        feed_dict[self.in_neigh_indices] = batch.in_neigh_indices
        feed_dict[self.in_neigh_edges] = batch.in_neigh_edges
        feed_dict[self.in_neigh_mask] = batch.in_neigh_mask

        feed_dict[self.out_neigh_indices] = batch.out_neigh_indices
        feed_dict[self.out_neigh_edges] = batch.out_neigh_edges
        feed_dict[self.out_neigh_mask] = batch.out_neigh_mask

        feed_dict[self.entity_indices] = batch.entity_indices
        feed_dict[self.entity_indices_mask] = batch.entity_indices_mask
        feed_dict[self.refs] = batch.y
        if is_train:
            return sess.run([self.accu, self.loss, self.train_op], feed_dict)
        else:
            return sess.run([self.accu, self.loss, self.output], feed_dict)


if __name__ == '__main__':
    summary = " Tokyo is the one of the biggest city in the world."
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
    print sentence_rouge(reference, summary)

