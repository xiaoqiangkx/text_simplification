"""Provide dataset"""
from os import remove, listdir
from os.path import exists
from nltk.corpus import stopwords



BASE = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/'
meta = BASE + 'meta/shard'
ncomp = BASE + 'ncomp/shard'
nsimp = BASE + 'nsimp/shard'
ncomp_map = BASE + 'ncomp_map/shard'
nsimp_map = BASE + 'nsimp_map/shard'
ppdb = BASE + 'ppdb/shard'
ppdb_rule = BASE + 'ppdb_rule/shard'

cnt = 0

stop_words_set = set(stopwords.words('english'))

ner_set = set()
for label in ['person', 'norp', 'fac', 'org', 'gpe', 'loc', 'product', 'event', 'work_of_art', 'law', 'language',
              'date', 'time', 'percent', 'money', 'quantity', 'ordinal', 'cardinal']:
    for i in range(0, 10):
        ner_set.add(label + str(i))


for shard_id in range(1280):
    if not exists(ncomp + str(shard_id)):
        continue
    file_meta = open(meta + str(shard_id))
    file_ncomp = open(ncomp + str(shard_id))
    file_nsimp = open(nsimp + str(shard_id))
    file_ncomp_map = open(ncomp_map + str(shard_id))
    file_nsimp_map = open(nsimp_map + str(shard_id))
    file_ppdb = open(ppdb + str(shard_id))
    file_ppdb_rule = open(ppdb_rule + str(shard_id))

    lines_meta, lines_ncomp, lines_nsimp, lines_ncomp_map, lines_nsimp_map, lines_ppdb, lines_ppdb_rule = (
        file_meta,
        file_ncomp,
        file_nsimp,
        file_ncomp_map,
        file_nsimp_map,
        file_ppdb,
        file_ppdb_rule)
    for line_meta, line_ncomp, line_nsimp, line_ncomp_map, line_nsimp_map, line_ppdb, line_ppdb_rule in zip(
            lines_meta, lines_ncomp, lines_nsimp, lines_ncomp_map, lines_nsimp_map, lines_ppdb, lines_ppdb_rule):
        # Check NE
        set_comp = set([w for w in line_ncomp.split() if w in ner_set])
        set_simp = set([w for w in line_nsimp.split() if w in ner_set])
        if set_comp != set_simp:
            continue

        items = line_meta.split('\t')
        # # Ignore same sentence
        if line_ncomp.strip() == line_nsimp.strip():
            continue

        if len(line_ppdb_rule.strip()) == 0:
            continue

        if float(line_ppdb) <= 0:
            continue

        extra_words = items[4].split()
        count_extra_words = 0
        for extra_word in extra_words:
            if extra_word not in stop_words_set and extra_word not in line_ppdb_rule:
                count_extra_words += 1

        if count_extra_words == 0:
            cnt += 1
        elif count_extra_words == 1:
            cnt += 1
            print(line_ncomp.strip())
            print(line_nsimp.strip())
            print(line_ppdb_rule.strip())
            print(line_ppdb.strip())
            print('=======')


print('CNT: %s' % cnt)

