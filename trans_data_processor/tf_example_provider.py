"""Create tensroflow.Example for model input."""
import tensorflow as tf
from os.path import exists
from datetime import datetime
from data_generator.vocab import Vocab
from multiprocessing import Pool

from util import constant
from util.data.text_encoder import SubwordTextEncoder
from model.model_config import WikiTransDefaultConfig



PATH_TRAIN = '/zfs1/hdaqing/saz31/dataset/trans_tf_example/ppdb_0/train.tfrecords.'
PATH_SUBVOCAB_COMP = '/zfs1/hdaqing/saz31/dataset/vocab/comp.subvocab'
PATH_SUBVOCAB_SIMP = '/zfs1/hdaqing/saz31/dataset/vocab/simp.subvocab'
PATH_PREFIX_COMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ncomp/shard'
PATH_PREFIX_SIMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/nsimp/shard'
PATH_PREFIX_PPDB = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb/shard'
PATH_PREFIX_PPDBRULE = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ppdb_rule/shard'
PATH_PREFIX_META = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/meta/shard'


# PATH_TRAIN = '/Users/sanqiangzhao/git/train.tfrecords.'
# PATH_SUBVOCAB_COMP = '/Users/sanqiangzhao/git/comp.subvocab'
# PATH_SUBVOCAB_SIMP = '/Users/sanqiangzhao/git/simp.subvocab'
# PATH_PREFIX_COMP = '/Users/sanqiangzhao/git/text_simplification_data/huge/pncomp/shard'
# PATH_PREFIX_SIMP = '/Users/sanqiangzhao/git/text_simplification_data/huge/pnsimp/shard'
# PATH_PREFIX_PPDB = '/Users/sanqiangzhao/git/text_simplification_data/huge/ppdb/shard'

# PATH_TRAIN = '/Users/sanqiangzhao/git/train.tfrecords.'
# PATH_SUBVOCAB_COMP = '/Users/sanqiangzhao/git/comp.subvocab'
# PATH_SUBVOCAB_SIMP = '/Users/sanqiangzhao/git/simp.subvocab'
# PATH_PREFIX_COMP = '/Users/sanqiangzhao/git/text_simplification_data/huge/pncomp/shard'
# PATH_PREFIX_SIMP = '/Users/sanqiangzhao/git/text_simplification_data/huge/pnsimp/shard'
# PATH_PREFIX_PPDB = '/Users/sanqiangzhao/git/text_simplification_data/huge/ppdb/shard'


subword_comp = SubwordTextEncoder(PATH_SUBVOCAB_COMP)
subword_simp = SubwordTextEncoder(PATH_SUBVOCAB_SIMP)
config = WikiTransDefaultConfig()
subword_comp_ex = Vocab(config, PATH_SUBVOCAB_COMP)
subword_simp_ex = Vocab(config, PATH_SUBVOCAB_SIMP)



# Validate Section method
from nltk.corpus import stopwords

stop_words_set = set(stopwords.words('english'))

ner_set = set()
for label in ['person', 'norp', 'fac', 'org', 'gpe', 'loc', 'product', 'event', 'work_of_art', 'law', 'language',
              'date', 'time', 'percent', 'money', 'quantity', 'ordinal', 'cardinal']:
    for i in range(0, 10):
        ner_set.add(label + str(i))
def _validate(line_ncomp, line_nsimp, ppdb_score, line_ppdb_rule, line_meta):
    set_comp = set([w for w in line_ncomp.split() if w in ner_set])
    set_simp = set([w for w in line_nsimp.split() if w in ner_set])
    if set_comp != set_simp:
        return False

    # # Ignore same sentence
    if line_ncomp.strip() == line_nsimp.strip():
        return False

    if len(line_ppdb_rule.strip()) == 0:
        return False

    if ppdb_score <= 0:
        return False

    count_extra_words = 0
    items = line_meta.split('\t')
    assert len(items) == 4 or len(items) == 5
    if len(items) == 5:
        extra_words = items[4].split()
        for extra_word in extra_words:
            if extra_word not in stop_words_set and extra_word not in line_ppdb_rule:
                count_extra_words += 1

    if count_extra_words == 0:
        return True
    return False



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


def generate_example(shard_id):
    if not exists(PATH_PREFIX_COMP + str(shard_id)) or not exists(PATH_PREFIX_SIMP + str(shard_id)):
        return
    writer = tf.python_io.TFRecordWriter(PATH_TRAIN + str(shard_id))
    print('Start shard id %s' % shard_id)
    s_time = datetime.now()
    lines_comp = open(PATH_PREFIX_COMP + str(shard_id))
    lines_simp = open(PATH_PREFIX_SIMP + str(shard_id))
    lines_ppdb = open(PATH_PREFIX_PPDB + str(shard_id))
    lines_ppdb_rule = open(PATH_PREFIX_PPDBRULE + str(shard_id))
    lines_meta = open(PATH_PREFIX_META + str(shard_id))
    for line_comp, line_simp, line_ppdb, line_ppdb_rule, line_meta in zip(
            lines_comp, lines_simp, lines_ppdb, lines_ppdb_rule, lines_meta):
        line_comp = line_comp.strip()
        line_simp = line_simp.strip()
        ppdb_score = [float(line_ppdb)]
        line_ppdb_rule = line_ppdb_rule.strip()
        line_meta = line_meta.strip()

        if not _validate(line_comp, line_simp, ppdb_score[0], line_ppdb_rule, line_meta):
            continue

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
    print('Finished shard id %s' % shard_id)
    time_span = datetime.now() - s_time
    print('Done id:%s with time span %s' % (shard_id, time_span))
    writer.close()

if __name__ == '__main__':
    p = Pool(10)
    p.map(generate_example, range(1280))