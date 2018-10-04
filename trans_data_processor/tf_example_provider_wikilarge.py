"""Create tensroflow.Example for model input specifically for wikilarge."""
import tensorflow as tf
from os.path import exists
from datetime import datetime
from data_generator.vocab import Vocab
from multiprocessing import Pool
from ppdb_util import populate_ppdb
from pppdb_features import get_score

from util import constant
from util.data.text_encoder import SubwordTextEncoder
from model.model_config import WikiTransDefaultConfig



PATH_TRAIN = '/Users/sanqiangzhao/git/text_simplification_data/train/wikilarge/train.tfrecords'
PATH_SUBVOCAB_COMP = '/Users/sanqiangzhao/git/text_simplification_data/vocab/comp.subvocab'
PATH_SUBVOCAB_SIMP = '/Users/sanqiangzhao/git/text_simplification_data/vocab/simp.subvocab'
PATH_PREFIX_COMP = '/Users/sanqiangzhao/git/text_simplification_data/train/wikilarge/words_comps'
PATH_PREFIX_SIMP = '/Users/sanqiangzhao/git/text_simplification_data/train/wikilarge/words_simps'


subword_comp = SubwordTextEncoder(PATH_SUBVOCAB_COMP)
subword_simp = SubwordTextEncoder(PATH_SUBVOCAB_SIMP)
config = WikiTransDefaultConfig()
subword_comp_ex = Vocab(config, PATH_SUBVOCAB_COMP)
subword_simp_ex = Vocab(config, PATH_SUBVOCAB_SIMP)


def _pad_length(ids, pad_id, max_len):
    if len(ids) < max_len:
        num_pad = max_len - len(ids)
        ids.extend(num_pad * pad_id)
    else:
        ids = ids[:max_len]
    return ids


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def generate_example():
    mapper = populate_ppdb()
    writer = tf.python_io.TFRecordWriter(PATH_TRAIN)
    lines_comp = open(PATH_PREFIX_COMP )
    lines_simp = open(PATH_PREFIX_SIMP)
    for line_comp, line_simp in zip(lines_comp, lines_simp):
        line_comp = line_comp.strip()
        line_simp = line_simp.strip()

        if len(line_comp.split()) <= 5 or len(line_simp.split()) <= 5:
            continue

        score, _ = get_score(line_comp, line_simp, mapper)
        ppdb_score = []
        line_comp_ids = subword_comp.encode(
            constant.SYMBOL_START + ' ' + line_comp.lower() + ' ' + constant.SYMBOL_END)
        line_comp_ids = _pad_length(line_comp_ids, subword_comp.encode(constant.SYMBOL_PAD), 210)
        line_simp_ids = subword_simp.encode(
            constant.SYMBOL_START + ' ' + line_simp.lower() + ' ' + constant.SYMBOL_END)
        line_simp_ids = _pad_length(line_simp_ids, subword_simp.encode(constant.SYMBOL_PAD), 200)

        feature = {
            'line_comp_ids': _int64_feature(line_comp_ids),
            'line_simp_ids': _int64_feature(line_simp_ids),
            'ppdb_score': _float_features(ppdb_score)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    generate_example()