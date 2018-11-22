import copy as cp
import random as rd
import tensorflow as tf
import glob

import numpy as np
from nltk import word_tokenize
from copy import deepcopy
import time
import random as rd

from data_generator.vocab import Vocab
from data_generator.rule import Rule
from util import constant
from data_generator import data_utils


# Deprecated: use Tf.Example (TfExampleTrainDataset) instead
class TrainData:
    """Fetching training dataset from plain data."""

    def __init__(self, model_config):
        self.model_config = model_config

        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        if self.model_config.subword_vocab_size > 0:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
            vocab_all_path = self.model_config.subword_vocab_all

        data_simple_path = self.model_config.train_dataset_simple
        data_complex_path = self.model_config.train_dataset_complex

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.vocab_simple = Vocab(model_config, vocab_simple_path)
            self.vocab_complex = Vocab(model_config, vocab_complex_path)
        elif (self.model_config.tie_embedding == 'all' or
                    self.model_config.tie_embedding == 'enc_dec'):
            self.vocab_simple = Vocab(model_config, vocab_all_path)
            self.vocab_complex = Vocab(model_config, vocab_all_path)

        self.size = self.get_size(data_complex_path)
        # Populate basic complex simple pairs
        if not self.model_config.it_train:
            self.data = self.populate_data(data_complex_path, data_simple_path,
                                           self.vocab_complex, self.vocab_simple, True)
        else:
            self.data_it = self.get_data_sample_it(data_simple_path, data_complex_path)

        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d.'
              % (data_simple_path, data_complex_path, self.size))

        if 'rule' in self.model_config.memory or 'rule' in self.model_config.rl_configs:
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)
            self.rules_target, self.rules_align = self.populate_rules(
                self.model_config.train_dataset_complex_ppdb, self.vocab_rule)
            assert len(self.rules_align) == self.size
            assert len(self.rules_target) == self.size
            print('Populate Rule with size:%s' % self.vocab_rule.get_rule_size())
        if model_config.pretrained:
            self.init_pretrained_embedding()

    # def process_line(self, line, vocab, max_len, need_raw=False):
    #     if self.model_config.tokenizer == 'split':
    #         words = line.split()
    #     elif self.model_config.tokenizer == 'nltk':
    #         words = word_tokenize(line)
    #     else:
    #         raise Exception('Unknown tokenizer.')
    #
    #     words = [Vocab.process_word(word, self.model_config)
    #              for word in words]
    #     if need_raw:
    #         words_raw = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    #     else:
    #         words_raw = None
    #
    #     if self.model_config.subword_vocab_size > 0:
    #         words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    #         words = vocab.encode(' '.join(words))
    #     else:
    #         words = [vocab.encode(word) for word in words]
    #         words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
    #                  [self.vocab_simple.encode(constant.SYMBOL_END)])
    #
    #     if self.model_config.subword_vocab_size > 0:
    #         pad_id = vocab.encode(constant.SYMBOL_PAD)
    #     else:
    #         pad_id = [vocab.encode(constant.SYMBOL_PAD)]
    #
    #     if len(words) < max_len:
    #         num_pad = max_len - len(words)
    #         words.extend(num_pad * pad_id)
    #     else:
    #         words = words[:max_len]
    #
    #     return words, words_raw

    def get_size(self, data_complex_path):
        return len(open(data_complex_path, encoding='utf-8').readlines())

    def get_data_sample_it(self, data_simple_path, data_complex_path):
        f_simple = open(data_simple_path, encoding='utf-8')
        f_complex = open(data_complex_path, encoding='utf-8')
        i = 0
        while True:
            if i >= self.size:
                f_simple = open(data_simple_path, encoding='utf-8')
                f_complex = open(data_complex_path, encoding='utf-8')
                i = 0
            line_complex = f_complex.readline()
            line_simple = f_simple.readline()
            if rd.random() < 0.5 or i >= self.size:
                i += 1
                continue

            words_complex, words_raw_comp = data_utils.process_line(
                line_complex, self.vocab_complex, self.model_config.max_complex_sentence, self.model_config, True)
            words_simple, words_raw_simp = data_utils.process_line(
                line_simple, self.vocab_simple, self.model_config.max_simple_sentence, self.model_config, True)

            supplement = {}
            if 'rule' in self.model_config.memory:
                supplement['rules_target'] = self.rules_target[i]
                supplement['rules_align'] = self.rules_align[i]

            obj = {}
            obj['words_comp'] = words_complex
            obj['words_simp'] = words_simple
            obj['words_raw_comp'] = words_raw_comp
            obj['words_raw_simp'] = words_raw_simp

            yield i, obj, supplement

            i += 1

    def populate_rules(self, rule_path, vocab_rule):
        data_target, data_align = [], []
        for line in open(rule_path, encoding='utf-8'):
            cur_rules = line.split('\t')
            tmp, tmp_align = [], []
            for cur_rule in cur_rules:
                rule_id, rule_origins, rule_targets = vocab_rule.encode(cur_rule)
                if rule_targets is not None and rule_origins is not None:
                    tmp.append((rule_id, [self.vocab_simple.encode(rule_target) for rule_target in rule_targets]))

                    if len(rule_origins) == 1 and len(rule_targets) == 1:
                        tmp_align.append(
                            (self.vocab_complex.encode(rule_origins[0]),
                             self.vocab_simple.encode(rule_targets[0])))
            data_target.append(tmp)
            data_align.append(tmp_align)

        return data_target, data_align

    def populate_data(self, data_path_comp, data_path_simp, vocab_comp, vocab_simp, need_raw=False):
        # Populate data into memory
        data = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
        lines_comp = open(data_path_comp, encoding='utf-8').readlines()
        lines_simp = open(data_path_simp, encoding='utf-8').readlines()
        assert len(lines_comp) == len(lines_simp)
        for line_id in range(len(lines_comp)):
            obj = {}
            line_comp = lines_comp[line_id]
            line_simp = lines_simp[line_id]
            words_comp, words_raw_comp = data_utils.process_line(
                line_comp, vocab_comp, self.model_config.max_complex_sentence, self.model_config, need_raw)
            words_simp, words_raw_simp = data_utils.process_line(
                line_simp, vocab_simp, self.model_config.max_simple_sentence, self.model_config, need_raw)
            obj['words_comp'] = words_comp
            obj['words_simp'] = words_simp
            if need_raw:
                obj['words_raw_comp'] = words_raw_comp
                obj['words_raw_simp'] = words_raw_simp

            data.append(obj)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        supplement = {}
        if 'rule' in self.model_config.memory:
            supplement['rules_target'] = self.rules_target[i]
            supplement['rules_align'] = self.rules_align[i]

        return i, self.data[i], supplement

    def init_pretrained_embedding(self):
        if self.model_config.subword_vocab_size > 0:
            # Subword doesn't need pretrained embedding.
            return

        if self.model_config.pretrained_embedding is None:
            return

        print('Use Pretrained Embedding\t%s.' % self.model_config.pretrained_embedding)

        if not hasattr(self, 'glove'):
            self.glove = {}
            for line in open(self.model_config.pretrained_embedding, encoding='utf-8'):
                pairs = line.split()
                word = ' '.join(pairs[:-self.model_config.dimension])
                if word in self.vocab_simple.w2i or word in self.vocab_complex.w2i:
                    embedding = pairs[-self.model_config.dimension:]
                    self.glove[word] = embedding

            # For vocabulary complex
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_complex = np.empty(
                (self.vocab_complex.vocab_size(), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_complex.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])

                    self.pretrained_emb_complex[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_complex[wid, :] = n_vector
                    random_cnt += 1
            assert self.vocab_complex.vocab_size() == random_cnt + pretrained_cnt
            print(
                'For Vocab Complex, %s words initialized with pretrained vector, '
                'other %s words initialized randomly.' %
                (pretrained_cnt, random_cnt))

            # For vocabulary simple
            pretrained_cnt = 0
            random_cnt = 0
            self.pretrained_emb_simple = np.empty(
                (len(self.vocab_simple.i2w), self.model_config.dimension), dtype=np.float32)
            for wid, word in enumerate(self.vocab_simple.i2w):
                if word in self.glove:
                    n_vector = np.array(self.glove[word])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    pretrained_cnt += 1
                else:
                    n_vector = np.array([np.random.uniform(-0.08, 0.08)
                                         for _ in range(self.model_config.dimension)])
                    self.pretrained_emb_simple[wid, :] = n_vector
                    random_cnt += 1
            assert len(self.vocab_simple.i2w) == random_cnt + pretrained_cnt
            print(
                'For Vocab Simple, %s words initialized with pretrained vector, '
                'other %s words initialized randomly.' %
                (pretrained_cnt, random_cnt))

            del self.glove


class TfExampleTrainDataset():
    """Fetching training dataset from tf.example Dataset"""

    def __init__(self, model_config):
        self.model_config = model_config

        if self.model_config.subword_vocab_size:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
        else:
            vocab_simple_path = self.model_config.vocab_simple
            vocab_complex_path = self.model_config.vocab_complex
        self.vocab_simple = Vocab(model_config, vocab_simple_path)
        self.vocab_complex = Vocab(model_config, vocab_complex_path)

        self.feature_set = {
            'line_comp': tf.FixedLenFeature([], tf.string),
            'line_simp': tf.FixedLenFeature([], tf.string),
        }
        if self.model_config.tune_style:
            self.feature_set['ppdb_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['len_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['add_score'] = tf.FixedLenFeature([], tf.float32)
            self.feature_set['dsim_score'] = tf.FixedLenFeature([], tf.float32)

        self.dataset = self._get_dataset(glob.glob(self.model_config.train_dataset))
        self.iterator = tf.data.Iterator.from_structure(
            self.dataset.output_types,
            self.dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.dataset)

        if self.model_config.dmode == 'alter':
            self.dataset2 = self._get_dataset(glob.glob(self.model_config.train_dataset2))
            self.iterator2 = tf.data.Iterator.from_structure(
                self.dataset2.output_types,
                self.dataset2.output_shapes)
            self.training_init_op2 = self.iterator2.make_initializer(self.dataset)

    def get_data_sample(self):
        if rd.random() >= 0.5 or self.model_config.dmode != 'alter':
            return self.iterator.get_next()
        else:
            return self.iterator2.get_next()

    def _parse(self, serialized_example):
        features = tf.parse_single_example(serialized_example, features=self.feature_set)

        def process_line_pair(line_complex, line_simple):
            words_complex, _ = data_utils.process_line(
                line_complex, self.vocab_complex, self.model_config.max_complex_sentence, self.model_config, True)
            words_simple, _ = data_utils.process_line(
                line_simple, self.vocab_simple, self.model_config.max_simple_sentence, self.model_config, True)
            return np.array(words_complex, np.int32), np.array(words_simple, np.int32)

        output_complex, output_simple = tf.py_func(
            process_line_pair,
            [features['line_comp'], features['line_simp']],
            [tf.int32, tf.int32])
        output_complex.set_shape(
            [self.model_config.max_complex_sentence])
        output_simple.set_shape(
            [self.model_config.max_simple_sentence])
        output =  {
            'line_comp_ids': output_complex,
            'line_simp_ids': output_simple,
        }

        if self.model_config.tune_style[0]:
            output['ppdb_score'] = features['ppdb_score']
        if self.model_config.tune_style[1]:
            output['dsim_score'] = features['dsim_score']
        if self.model_config.tune_style[2]:
            output['add_score'] = features['add_score']
        if self.model_config.tune_style[3]:
            output['len_score'] = features['len_score']
        return output

    def _get_dataset(self, path):
        dataset = tf.data.TFRecordDataset([path]).repeat().shuffle(10000)
        dataset = dataset.map(self._parse, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=10000)
        return dataset.batch(self.model_config.batch_size)
