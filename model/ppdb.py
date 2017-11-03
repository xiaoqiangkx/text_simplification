from model.model_config import get_path, WikiDressLargeDefault
from data_generator.vocab import Vocab
from collections import defaultdict
from model.model_config import DefaultTrainConfig, WikiDressLargeTrainConfig

from util.arguments import get_args
import random as rd
from numpy.random import choice
from nltk.stem import WordNetLemmatizer
import copy as cp
import re
import time


args = get_args()


class PPDB:
    def __init__(self, model_config):
        self.model_config = model_config
        self.wnl = WordNetLemmatizer()
        # rules is in structure
        # ori_words => tag => list of [tar words, its weight]
        self.rules = defaultdict(dict)
        self.populate_ppdb()

    def populate_ppdb(self):
        for line in open(self.model_config.path_ppdb_refine, encoding='utf-8'):
            items = line.split('\t')
            weight = float(items[1])
            ori_words = items[3].strip()
            tar_words = items[4].strip()
            tags = items[2]
            for tag in tags.split('\\'):
                # Tag in www.cs.cornell.edu/courses/cs474/2004fa/lec1.pdf
                if tag[0] == '[':
                    tag = tag[1:]
                if tag[-1] == ']':
                    tag = tag[:-1]

                if re.match('[a-zA-Z]+', ori_words) or re.match('[a-zA-Z]+', tar_words):
                    if ori_words not in self.rules:
                        self.rules[ori_words] = {}
                    if tag not in self.rules[ori_words]:
                        self.rules[ori_words][tag] = {}
                    if tar_words in self.rules[ori_words][tag]:
                        self.rules[ori_words][tag][tar_words] = max(weight, self.rules[ori_words][tag][tar_words])
                    else:
                        self.rules[ori_words][tag][tar_words] = weight

    """generate_synatx and recursive_gen are through NLTK called Stanford NLP tools
       for faster processing, we use java. script/SyntaxParser.java"""
    # def generate_synatx(self, syntax):
    #     outout_list = []
    #     for tree in syntax:
    #         if tree.label() == 'ROOT':
    #             tree = tree[0]
    #         try:
    #             self.recursive_gen(tree, outout_list)
    #         except Exception:
    #             print(syntax)
    #     outout_list = '\t'.join(outout_list)
    #     return outout_list
    #
    # def recursive_gen(self, tree, outout_list):
    #     if type(tree[0]) == str and len(tree) == 1:
    #         word = tree[0]
    #         if '@' not in word:
    #             label = tree.label()
    #             outout_list.append('%s=>%s' % (label, word))
    #         return word
    #
    #     output = []
    #     for node in tree:
    #         node_str = self.recursive_gen(node, outout_list)
    #         output.append(node_str)
    #     words = ' '.join(output)
    #     if '@' not in words:
    #         label = tree.label()
    #         outout_list.append('%s=>%s' % (label, words))
    #     return words

    def process_training(self, line_sep='\n', is_comp=False):
        output = ''
        line_idx = 0
        if is_comp:
            syntaxs = open(self.model_config.train_dataset_complex_syntax, encoding='utf-8').readlines()
            sents_simp = open(self.model_config.train_dataset_simple, encoding='utf-8').readlines()
            sents_comp = open(self.model_config.train_dataset_complex, encoding='utf-8').readlines()
        else:
            syntaxs = open(self.model_config.train_dataset_simple_syntax, encoding='utf-8').readlines()
            sents_simp, sents_comp = None, None
        pre_time = time.time()
        for i, syntax in enumerate(syntaxs):
            syntax = syntax.strip()
            rules = ''
            if len(syntax) > 0:
                syntax_pairs = [p.split('=>') for p in syntax.split('\t')]
                if is_comp:
                    rules = ppdb.process_sent(syntax_pairs, sents_simp[i], sents_comp[i])
                else:
                    rules = ppdb.process_sent(syntax_pairs)
            rules = '\t'.join(rules)
            output = line_sep.join([output, rules])
            line_idx += 1
            if line_idx % 1000 == 0:
                cur_time = time.time()
                print('processed %s. in %s' % (line_idx, cur_time - pre_time))
                pre_time = cur_time

        output = output[len(line_sep):]

        if is_comp:
            f = open(self.model_config.train_dataset_complex_ppdb, 'w', encoding='utf-8')
        else:
            f = open(self.model_config.train_dataset_simple_ppdb, 'w', encoding='utf-8')
        f.write(output)
        f.close()

    def process_sent(self, syntax_pairs, sent_simp=None, sent_comp=None):
        # PPDB doesn't have am is are but only be
        filter_rules = []

        if sent_simp is not None:
            sent_simp = sent_simp.lower().split()
            sent_comp = sent_comp.lower().split()

        def list_contain(tar_list, sent_list):
            if len(tar_list) == 0 or len(sent_list) == 0:
                return False
            for i in range(len(sent_list)):
                j = 0
                il = i
                while il < len(sent_list) and sent_list[il] == tar_list[j]:
                    j += 1
                    il += 1
                    if j == len(tar_list):
                        return True
            return False

        for tag, words in syntax_pairs:
            words = words.lower()
            words_list = words.split()
            for be in ['am', 'is', 'are']:
                while be in words_list:
                    be_idx = words_list.index(be)
                    words_list[be_idx] = 'be'
            words = ' '.join(words_list)
            if sent_comp is not None and sent_simp is not None:
                # Comp mode
                # For now, we use non-tag approach for maximize matching
                if words in self.rules:
                    # Non-tag case
                    for tag in self.rules[words]:
                        target_word_pairs = self.rules[words][tag]
                        for target_word in target_word_pairs:
                            target_word_list = target_word.split()
                            if (list_contain(target_word_list, sent_simp) and not list_contain(words_list, sent_simp)
                                and not list_contain(target_word_list, sent_comp)):
                                filter_rules.append('%s=>%s=>%s=>%s' % (tag, words, target_word, self.rules[words][tag][target_word]))
            # TODO(sanqiang): simp mode need to improved
            # else:
            #     # # Simp mode
            #     # if words in self.rules:
            #     #     if tag in self.rules[words]:
            #     #         filter_rules.append('%s=>%s' % (tag, words))
            #     #
            #     #     if 'X' in self.rules[words]:
            #     #         filter_rules.append('%s=>%s' % ('X', words))
            #     #
            #     #     if 'NEW' in self.rules[words]:
            #     #         filter_rules.append('%s=>%s' % ('X', words))

        return filter_rules

    def simplify(self, sent_simp, sent_comp,
                 pairs_simp, pairs_comp,
                 vocab_simple, vocab_complex, smooth_factor=1e-7):
        def isplural(word):
            lemma = self.wnl.lemmatize(word, 'n')
            plural = True if word is not lemma else False
            return plural, lemma

        def ori2be(sent, words):
            be = None
            rule_ori_list = words.split()
            if 'be' in rule_ori_list:
                be_idx = rule_ori_list.index('be')
                rule_ori_list[be_idx] = 'is'
                be = 'is'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'are'
                be = 'are'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'am'
                be = 'am'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                rule_ori_list[be_idx] = 'be'
                be = 'be'
                if ' '.join(rule_ori_list) in sent:
                    return ' '.join(rule_ori_list), be
                raise Exception('be parsing error: %s\n%s' % (sent, words))
            return sent, be

        def be2ori(sent, target, be):
            target_list = target.split()
            if 'be' in target_list:
                be_idx = target_list.index('be')
                if be is None:
                    sent_list = sent.split()
                    if 'are' in sent_list:
                        be = 'are'
                    elif 'am' in sent_list:
                        be = 'am'
                    elif 'is' in sent_list:
                        be = 'is'
                    else:
                        need_replaced = True
                        for word in sent.split():
                            if word == 'and' or isplural(word):
                                be = 'are'
                                need_replaced = False
                                break
                        if need_replaced:
                            be = 'is'
                target_list[be_idx] = be
                return ' '.join(target_list)
            return target

        prob_mapper = {}
        if 'comp' in self.model_config.ppdb_mode and pairs_comp:
            for pair_comp in pairs_comp:
                target_words = pair_comp[2]
                prob = float(pair_comp[3])
                for target_word in target_words.split():
                    if self.model_config.ppdb_args is None:
                        prob_mapper[target_word] = prob
                    else:
                        # Because the weight is supposed to be larger than 1 and it already increment 1 below
                        prob_mapper[target_word] = self.model_config.ppdb_args[0] - 1.0

        if self.model_config.ppdb_args is not None and len(self.model_config.ppdb_args) >= 2:
            sent_comp_wds = set([vocab_complex.describe(wid).lower() for wid in sent_comp
                                 if wid > config.REVERED_VOCAB_SIZE])
            sent_simp_wds = set([vocab_simple.describe(wid).lower() for wid in sent_simp
                                 if wid > config.REVERED_VOCAB_SIZE])
            for word in sent_comp_wds & sent_simp_wds:
                prob_mapper[word] = self.model_config.ppdb_args[1] - 1.0

        #TODO(sanqiang) Simp mode need to improved
        if 'simp' in self.model_config.ppdb_mode and pairs_simp:
            num_trials = 5
            while True:
                num_trials -= 1
                if num_trials == 0:
                    break

                idx = int(rd.random() * len(pairs_simp))
                pair = pairs_simp[idx]
                if len(pair) != 2:
                    print('pairs error! %s in %s.' % (pairs_simp, sent_simp))
                    continue
                words = pair[1]
                tag = pair[0]
                # Use originl (includes be) to check target
                target_pairs = cp.deepcopy(self.rules[words][tag])
                if 'X' in self.rules[words]:
                    target_pairs += self.rules[words]['X']
                # Use replaced one (be will changed to am,is,are,be) to check replace
                sent_simp, be = ori2be(sent_simp, words)

                rule_tars = [p[0] for p in target_pairs]
                rule_tars_p_norm = sum([p[1] + smooth_factor for p in target_pairs])
                rule_tars_p = [(p[1] + smooth_factor) / rule_tars_p_norm for p in target_pairs]
                if not rule_tars:
                    print('empty rule_tars')
                    continue
                target = choice(rule_tars, p=rule_tars_p, size=1)[0]
                simp_target_p = rule_tars_p[rule_tars.index(target)]

                target = be2ori(sent_simp, target, be)
                nsent = sent_simp.lower().replace(words, target)

                simp_target_list = target.split()
                # Check whether simplified phrase contain UNK
                for word in simp_target_list:
                    if not vocab_simple.contain(word):
                        continue
                sent_simp = nsent
                break

        nsent_idx = []
        nsent_weight = []
        for wid in sent_simp:
            word = vocab_simple.describe(wid)
            word = word.lower()
            nsent_idx.append(wid)
            if word in prob_mapper:
                nsent_weight.append(1.0 + prob_mapper[word])
            else:
                nsent_weight.append(1.0)

        return nsent_idx, nsent_weight


def get_refine_data():
    """Keep the rules that only relevant to our vocab so that speedup the processing."""
    model_config = WikiDressLargeDefault()
    vocab = Vocab(model_config,
                  get_path(
                      '../text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.vocab.lower'))

    path_ppdb = get_path(
        '../text_simplification_data/ppdb/SimplePPDB')
    path_ppdb_refine = get_path(
        '../text_simplification_data/ppdb/ppdb_rules2.simp.rules')

    noutputs = []
    line_idx = 0
    f_simp = open(path_ppdb_refine, 'w', encoding='utf-8')
    for line in open(path_ppdb, encoding='utf-8'):
        line_idx += 1
        if line_idx % 1000 == 0:
            print(line_idx)
        items = line.split('\t')
        tag = items[2]
        if tag == '[CD]':
            continue
        ori_words = items[3]
        tar_words = items[4]
        ori_words_list = ori_words.split()
        tar_words_list = tar_words.split()
        if ori_words != tar_words and (
                    any([word for word in ori_words_list if vocab.contain(word)]) or
                    any([word for word in tar_words_list if vocab.contain(word)])):
            noutputs.append(line.strip())

    for line in noutputs:
        f_simp.write(line)
        f_simp.write('\n')
    f_simp.flush()
    f_simp.close()


if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainConfig()
    # get_refine_data()
    ppdb = PPDB(config)
    ppdb.process_training(is_comp=True)
    # ppdb.simplify(
    #     'There is manuscript evidence that PERSON@1 continued to work on these pieces as late as the period NUMBER@1 â '' NUMBER@2 , and that her niece and nephew , PERSON@2 and PERSON@3 , made further additions as late as NUMBER@3 .',
    #     'manuscript	evidence that	continued	to work on	the period	niece	further')