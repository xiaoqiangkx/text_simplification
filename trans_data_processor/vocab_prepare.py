"""Provide vocabulary"""
from collections import Counter
from os.path import exists
from trans_data_processor.vocab_util import SubwordTextEncoder
from trans_data_processor import vocab_util

# PATH_PREFIX_COMP = '/home/zhaos5/tmp/pncomp/shard'
# PATH_PREFIX_SIMP = '/home/zhaos5/tmp/pnsimp/shard'
# PATH_VOCAB_COMP = '/home/zhaos5/tmp/comp.vocab'
# PATH_VOCAB_SIMP = '/home/zhaos5/tmp/simp.vocab'
# PATH_SUBVOCAB_COMP = '/home/zhaos5/tmp/comp.subvocab'
# PATH_SUBVOCAB_SIMP = '/home/zhaos5/tmp/simp.subvocab'

PATH_PREFIX_COMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/ncomp/shard'
PATH_PREFIX_SIMP = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/nsimp/shard'
PATH_VOCAB_COMP = '/zfs1/hdaqing/saz31/dataset/vocab/comp.vocab'
PATH_VOCAB_SIMP = '/zfs1/hdaqing/saz31/dataset/vocab/simp.vocab'
PATH_SUBVOCAB_COMP = '/zfs1/hdaqing/saz31/dataset/vocab/comp.subvocab'
PATH_SUBVOCAB_SIMP = '/zfs1/hdaqing/saz31/dataset/vocab/simp.subvocab'

c_comp, c_simp = Counter(), Counter()

for shard_id in range(1280):
    if not exists(PATH_PREFIX_COMP + str(shard_id)) or not exists(PATH_PREFIX_SIMP + str(shard_id)):
        continue
    lines_comp = open(PATH_PREFIX_COMP + str(shard_id))
    lines_simp = open(PATH_PREFIX_SIMP + str(shard_id))
    for line_comp, line_simp in zip(lines_comp, lines_simp):
        c_comp.update(line_comp.split())
        c_simp.update(line_simp.split())

vocab_comps = []
for w, c in c_comp.most_common():
    vocab_comps.append('%s\t%s' % (w, c))
open(PATH_VOCAB_COMP, 'w').write('\n'.join(vocab_comps))

vocab_simps = []
for w, c in c_simp.most_common():
    vocab_simps.append('%s\t%s' % (w, c))
open(PATH_VOCAB_SIMP, 'w').write('\n'.join(vocab_simps))

print('Created Vocab.')

sub_word_comp_feeder = {}
for line in open(PATH_VOCAB_COMP):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    sub_word_comp_feeder[word] = cnt

c_comp = Counter(sub_word_comp_feeder)
sub_word_comp = SubwordTextEncoder.build_to_target_size(
    8000, c_comp, 1, 1e5, num_iterations=10)
for i, subtoken_string in enumerate(sub_word_comp._all_subtoken_strings):
    if subtoken_string in vocab_util.RESERVED_TOKENS_DICT:
        sub_word_comp._all_subtoken_strings[i] = subtoken_string + "_"
sub_word_comp.store_to_file(PATH_SUBVOCAB_COMP)

sub_word_simp_feeder = {}
for line in open(PATH_VOCAB_SIMP):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    sub_word_simp_feeder[word] = cnt

c_simp = Counter(sub_word_simp_feeder)
sub_word_simp = SubwordTextEncoder.build_to_target_size(
    8000, c_simp, 1, 1e5, num_iterations=10)
for i, subtoken_string in enumerate(sub_word_simp._all_subtoken_strings):
    if subtoken_string in vocab_util.RESERVED_TOKENS_DICT:
        sub_word_simp._all_subtoken_strings[i] = subtoken_string + "_"
sub_word_simp.store_to_file(PATH_SUBVOCAB_SIMP)

print('Created Subvocab.')


