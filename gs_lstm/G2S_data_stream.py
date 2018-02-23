import json
import re
import codecs
import numpy as np
import random
import padding_utils

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines

def read_nary_file(inpath, options):
    all_words = []
    all_lemmas = []
    all_poses = []
    all_in_neigh = []
    all_in_label = []
    all_out_neigh = [] # [batch, node, neigh]
    all_out_label = [] # [batch, node, neigh]
    all_entity_indices = [] # [batch, 3, entity_size]
    all_y = []
    if options.class_num == 2:
        relation_set = {'resistance or non-response':0, 'sensitivity':0, 'response':0, 'resistance':0, 'None':1, }
    elif options.class_num == 5:
        relation_set = {'resistance or non-response':0, 'sensitivity':1, 'response':2, 'resistance':3, 'None':4, }
    else:
        assert False, 'Illegal class num'
    max_words = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_entity_size = 0
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for inst in json.load(f):
            words = []
            lemmas = []
            poses = []
            if options.only_single_sent and len(inst['sentences']) > 1:
                continue
            for sentence in inst['sentences']:
                for node in sentence['nodes']:
                    words.append(node['label'])
                    lemmas.append(node['lemma'])
                    poses.append(node['postag'])
            max_words = max(max_words, len(words))
            all_words.append(words)
            all_lemmas.append(lemmas)
            all_poses.append(poses)
            in_neigh = [[i,] for i,_ in enumerate(words)]
            in_label = [['self',] for i,_ in enumerate(words)]
            out_neigh = [[i,] for i,_ in enumerate(words)]
            out_label = [['self',] for i,_ in enumerate(words)]
            for sentence in inst['sentences']:
                for node in sentence['nodes']:
                    i = node['index']
                    for arc in node['arcs']:
                        j = arc['toIndex']
                        l = arc['label']
                        l = l.split('::')[0]
                        l = l.split('_')[0]
                        l = l.split('(')[0]
                        if j == -1 or l == '':
                            continue
                        in_neigh[j].append(i)
                        in_label[j].append(l)
                        out_neigh[i].append(j)
                        out_label[i].append(l)
            for _i in in_neigh:
                max_in_neigh = max(max_in_neigh, len(_i))
            for _o in out_neigh:
                max_out_neigh = max(max_out_neigh, len(_o))
            all_in_neigh.append(in_neigh)
            all_in_label.append(in_label)
            all_out_neigh.append(out_neigh)
            all_out_label.append(out_label)
            entity_indices = []
            for entity in inst['entities']:
                entity_indices.append(entity['indices'])
                max_entity_size = max(max_entity_size, len(entity['indices']))
            assert len(entity_indices) == options.entity_num
            all_entity_indices.append(entity_indices)
            all_y.append(relation_set[inst['relationLabel'].strip()])
    all_lex = all_lemmas if options.word_format == 'lemma' else all_words
    return zip(all_lex, all_poses, all_in_neigh, all_in_label, all_out_neigh, all_out_label, all_entity_indices, all_y), \
            max_words, max_in_neigh, max_out_neigh, max_entity_size


def read_nary_from_fof(fofpath, options):
    all_paths = read_text_file(fofpath)
    all_instances = []
    max_words = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_entity_size = 0
    for cur_path in all_paths:
        print(cur_path)
        cur_instances, cur_words, cur_in_neigh, cur_out_neigh, cur_entity_size = read_nary_file(cur_path, options)
        all_instances.extend(cur_instances)
        max_words = max(max_words, cur_words)
        max_in_neigh = max(max_in_neigh, cur_in_neigh)
        max_out_neigh = max(max_out_neigh, cur_out_neigh)
        max_entity_size = max(max_entity_size, cur_entity_size)
    return all_instances, max_words, max_in_neigh, max_out_neigh, max_entity_size


def collect_vocabs(all_instances):
    all_words = set()
    all_chars = set()
    all_edgelabels = set()
    for (lex, poses, in_neigh, in_label, out_neigh, out_label, entity_indices, y) in all_instances:
        all_words.update(lex)
        for l in lex:
            if l.isspace() == False: all_chars.update(l)
        for edges in in_label:
            all_edgelabels.update(edges)
        for edges in out_label:
            all_edgelabels.update(edges)
    return (all_words, all_chars, all_edgelabels)

class G2SDataStream(object):
    def __init__(self, all_instances, word_vocab=None, char_vocab=None, edgelabel_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.options = options
        if batch_size ==-1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (lex, poses, in_neigh, in_label, out_neigh, out_label, entity_indices, y) in all_instances:
            if options.max_node_num != -1 and len(lex) > options.max_node_num:
                continue # remove very long passages
            in_neigh = [x[:options.max_in_neigh_num] for x in in_neigh]
            in_label = [x[:options.max_in_neigh_num] for x in in_label]
            out_neigh = [x[:options.max_out_neigh_num] for x in out_neigh]
            out_label = [x[:options.max_out_neigh_num] for x in out_label]

            lex_idx = word_vocab.to_index_sequence_for_list(lex)
            lex_chars_idx = None
            if options.with_char:
                lex_chars_idx = char_vocab.to_character_matrix_for_list(lex, max_char_per_word=options.max_char_per_word)
            in_label_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in in_label]
            out_label_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in out_label]
            instances.append((lex_idx, lex_chars_idx, in_neigh, in_label_idx, out_neigh, out_label_idx, entity_indices, y))

        all_instances = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda inst: len(inst[0]))
        if isShuffle:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = G2SBatch(cur_instances, options, word_vocab=word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class G2SBatch(object):
    def __init__(self, instances, options, word_vocab=None):
        self.options = options

        self.instances = instances # list of tuples
        self.batch_size = len(instances)
        self.vocab = word_vocab

        # node num
        self.node_num = [] # [batch_size]
        for (lex_idx, lex_chars_idx, in_neigh, in_label_idx, out_neigh, out_label_idx, entity_indices, y) in instances:
            self.node_num.append(len(lex_idx))
        self.node_num = np.array(self.node_num, dtype=np.int32)

        # node char num
        if options.with_char:
            self.nodes_chars_num = [[len(lex_chars_idx) for lex_chars_idx in instance[1]] for instance in instances]
            self.nodes_chars_num = padding_utils.pad_2d_vals_no_size(self.nodes_chars_num)

        # neigh mask
        self.in_neigh_mask = [] # [batch_size, node_num, neigh_num]
        self.out_neigh_mask = []
        self.entity_indices_mask = []
        for instance in instances:
            ins = []
            for in_neighs in instance[2]:
                ins.append([1 for _ in in_neighs])
            self.in_neigh_mask.append(ins)
            outs = []
            for out_neighs in instance[4]:
                outs.append([1 for _ in out_neighs])
            self.out_neigh_mask.append(outs)
            idxs = []
            for entity_indices in instance[6]:
                idxs.append([1 for _ in entity_indices])
            self.entity_indices_mask.append(idxs)
        self.in_neigh_mask = padding_utils.pad_3d_vals_no_size(self.in_neigh_mask)
        self.out_neigh_mask = padding_utils.pad_3d_vals_no_size(self.out_neigh_mask)
        self.entity_indices_mask = padding_utils.pad_3d_vals_no_size(self.entity_indices_mask)

        # the actual contents
        self.nodes = [x[0] for x in instances]
        if options.with_char:
            self.nodes_chars = [x[1] for x in instances] # [batch_size, sent_len, char_num]
        self.in_neigh_indices = [x[2] for x in instances]
        self.in_neigh_edges = [x[3] for x in instances]
        self.out_neigh_indices = [x[4] for x in instances]
        self.out_neigh_edges = [x[5] for x in instances]
        self.entity_indices = [x[6] for x in instances]
        self.y = [x[7] for x in instances]

        # making ndarray
        self.nodes = padding_utils.pad_2d_vals_no_size(self.nodes)
        if options.with_char:
            self.nodes_chars = padding_utils.pad_3d_vals_no_size(self.nodes_chars)
        self.in_neigh_indices = padding_utils.pad_3d_vals_no_size(self.in_neigh_indices)
        self.in_neigh_edges = padding_utils.pad_3d_vals_no_size(self.in_neigh_edges)
        self.out_neigh_indices = padding_utils.pad_3d_vals_no_size(self.out_neigh_indices)
        self.out_neigh_edges = padding_utils.pad_3d_vals_no_size(self.out_neigh_edges)
        self.entity_indices = padding_utils.pad_3d_vals_no_size(self.entity_indices)
        self.y = np.asarray(self.y, dtype='int32')

        assert self.in_neigh_mask.shape == self.in_neigh_indices.shape
        assert self.in_neigh_mask.shape == self.in_neigh_edges.shape
        assert self.out_neigh_mask.shape == self.out_neigh_indices.shape
        assert self.out_neigh_mask.shape == self.out_neigh_edges.shape
        assert self.entity_indices_mask.shape == self.entity_indices.shape

        assert self.entity_indices.shape[1] == options.entity_num
        assert self.entity_indices_mask.shape[1] == options.entity_num

    def get_amrside_anonyids(self, anony_ids):
        assert self.batch_size == 1 # only for beam search
        if self.options.__dict__.has_key("enc_word_vec_path"):
            assert self.options.enc_word_vec_path == self.options.dec_word_vec_path # only when enc_vocab == dec_vocab
        self.amr_anony_ids = set(self.instances[0][0]) & anony_ids # sent1 of inst_0


if __name__ == "__main__":
    all_instances, max_node_num, max_in_neigh_num, max_out_neigh_num, max_entity_size = read_nary_from_fof(
            './data/data_list', 'lemma')
    print sum(len(x[0]) for x in all_instances)/len(all_instances)
    print(max_in_neigh_num)
    print(max_out_neigh_num)
    print(max_node_num)
    print(max_entity_size)
    print('DONE!')

