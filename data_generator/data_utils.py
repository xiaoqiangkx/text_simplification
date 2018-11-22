from data_generator.vocab import Vocab
from nltk import word_tokenize
from util import constant


def process_line(line, vocab, max_len, model_config, need_raw=False, lower_case=True):
    if lower_case:
        line = line.lower()
    if type(line) == bytes:
        line = str(line, 'utf-8')

    if model_config.tokenizer == 'split':
        words = line.split()
    elif model_config.tokenizer == 'nltk':
        words = word_tokenize(line)
    else:
        raise Exception('Unknown tokenizer.')

    words = [Vocab.process_word(word, model_config)
             for word in words]
    if need_raw:
        words_raw = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
    else:
        words_raw = None

    if model_config.subword_vocab_size > 0:
        words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
        words = vocab.encode(' '.join(words))
    else:
        words = [vocab.encode(word) for word in words]
        words = ([vocab.encode(constant.SYMBOL_START)] + words +
                 [vocab.encode(constant.SYMBOL_END)])

    if model_config.subword_vocab_size > 0:
        pad_id = vocab.encode(constant.SYMBOL_PAD)
    else:
        pad_id = [vocab.encode(constant.SYMBOL_PAD)]

    if len(words) < max_len:
        num_pad = max_len - len(words)
        words.extend(num_pad * pad_id)
    else:
        words = words[:max_len]

    return words, words_raw