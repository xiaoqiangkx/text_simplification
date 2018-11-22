# Deprecated: useless, spacy ner does not needed
"""Multi process for processing the trans data"""
import spacy
from collections import defaultdict
from datetime import datetime
from os.path import exists
from os import mkdir
from multiprocessing import Pool


nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])

is_trans = True

if is_trans:
    OPATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/'
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/'
    NPATH_SPACY_NER_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_trans/ner/spacy_ner'
else:
    OPATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/'
    PATH_PREFIX = '/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/ner/'
    NPATH_SPACY_NER_PREFIX = '/zfs1/hdaqing/saz31/tmp_wikilarge/tmp_trans/ner/spacy_ner'


def preprocess_line(line_comp, line_simp, line_ner_comp, line_ner_simp):
    """Generate spacy ner and do the replacement."""
    output_mapper = []
    doc_comp, doc_simp = nlp(line_comp), nlp(line_simp)

    mapper = {}
    label_counter = defaultdict(int)
    for ent in doc_comp.ents:
        ent_label = ent.label_ + str(label_counter[ent.label_]) + 's'
        line_comp = line_comp.replace(ent.text, ent_label)
        mapper[ent.text] = ent_label
        label_counter[ent.label_] += 1

    for ent in doc_simp.ents:
        ent_label = ent.label_ + str(label_counter[ent.label_]) + 's'
        line_simp = line_simp.replace(ent.text, ent_label)
        mapper[ent.text] = ent_label
        label_counter[ent.label_] += 1

    for ent_text in mapper:
        if ent_text in line_ner_comp and ent_text in line_ner_simp:
            line_ner_comp = line_ner_comp.replace(ent_text, mapper[ent_text])
            line_ner_simp = line_ner_simp.replace(ent_text, mapper[ent_text])
            output_mapper.append('%s=>%s' % (ent_text, mapper[ent_text]))

    return line_ner_comp, line_ner_simp, '\t'.join(output_mapper)


def preprocess_file(path_comp, path_simp, path_ner_comp, path_ner_simp, path_map):
    """Generate ner mapper and tokenized str for path."""
    s_time = datetime.now()
    words_comps, words_simps, mappers = [], [], []
    lines_comp = open(path_comp).readlines()
    lines_simp = open(path_simp).readlines()
    lines_ner_comp = open(path_ner_comp).readlines()
    lines_ner_simp = open(path_ner_simp).readlines()
    for line_comp, line_simp, line_ner_comp, line_ner_simp in zip(lines_comp, lines_simp, lines_ner_comp, lines_ner_simp):
        words_comp, words_simp, mapper = preprocess_line(
            line_comp, line_simp, line_ner_comp, line_ner_simp)
        words_comps.append(words_comp)
        words_simps.append(words_simp)
        mappers.append(mapper)

        if len(words_comps) % 100 == 0:
            time_span = datetime.now() - s_time
            print('Done line %s with time span %s' % (len(words_comps), time_span))
            s_time = datetime.now()

    return words_comps, words_simps, mappers


def process_trans(id):

    if not exists(OPATH_PREFIX + '/ncomp/shard%s' % id) or not exists(OPATH_PREFIX + '/nsimp/shard%s' % id):
        return

    if (exists(PATH_PREFIX + '/ncomp2/shard%s' % id) and
        exists(PATH_PREFIX + '/nsimp2/shard%s' % id) and
        exists(NPATH_SPACY_NER_PREFIX + '/shard%s' % id)):
        return

    f_ncomp = open(PATH_PREFIX + '/ncomp2/shard%s' % id, 'w')
    f_nsimp = open(PATH_PREFIX + '/nsimp2/shard%s' % id, 'w')
    f_map = open(NPATH_SPACY_NER_PREFIX + '/shard%s' % id, 'w')

    words_comps, words_simps, mappers = preprocess_file(
        OPATH_PREFIX + '/ncomp/shard%s' % id,
        OPATH_PREFIX + '/nsimp/shard%s' % id,
        PATH_PREFIX + '/ncomp/shard%s' % id,
        PATH_PREFIX + '/nsimp/shard%s' % id,
        NPATH_SPACY_NER_PREFIX + '/shard%s' % id)

    print('Start id:%s' % id)
    s_time = datetime.now()

    f_ncomp.write('\n'.join(words_comps))
    f_nsimp.write('\n'.join(words_simps))
    f_map.write('\n'.join(mappers))
    time_span = datetime.now() - s_time
    print('Done id:%s with time span %s' % (id, time_span))


if __name__ == '__main__':
    p = Pool(4)
    p.map(process_trans, range(1280))
