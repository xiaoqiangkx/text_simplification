import os
from util.arguments import get_args


args = get_args()


def get_path(file_path, env='sys'):
    if env == 'crc':
        return "/zfs1/hdaqing/saz31/text_simplification_0924/tmp/" + file_path
    elif env == 'aws':
        return '/home/zhaos5/ts/perf/' + file_path
    else:
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


class DefaultConfig():
    environment = args.environment
    train_mode = args.train_mode
    num_gpus = args.num_gpus
    framework = args.framework
    warm_start = args.warm_start
    warm_config = args.warm_config.split(':')
    use_partial_restore = args.use_partial_restore
    batch_size = 3
    dimension = 50
    max_complex_sentence = 10
    max_simple_sentence = 8
    model_eval_freq = args.model_eval_freq
    it_train = args.it_train
    model_print_freq = 10
    save_model_secs = 600
    number_samples = args.number_samples
    dmode = args.dmode

    min_count = 0
    lower_case = args.lower_case
    tokenizer = 'split' # split: white space split / nltk: nltk tokenizer

    optimizer = args.optimizer
    learning_rate_warmup_steps = 50000
    learning_rate = args.learning_rate
    max_grad_norm = 4.0
    layer_prepostprocess_dropout = args.layer_prepostprocess_dropout

    beam_search_size = -1

    # Overwrite transformer config
    # timing: use positional encoding
    hparams_pos = args.hparams_pos

    num_heads = args.num_heads
    num_hidden_layers = args.num_hidden_layers
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers

    # post process
    replace_unk_by_attn = False
    replace_unk_by_emb = False
    replace_unk_by_cnt = False
    replace_ner = True
    if framework == 'transformer':
        replace_unk_by_emb = True
    elif framework == 'seq2seq':
        replace_unk_by_attn = False


    # deprecated: std of trunc norm init, used for initializing embedding / w
    # trunc_norm_init_std = 1e-4

    # tie_embedding configuration description
    # all:encoder/decoder/output; dec_out: decoder/output; enc_dec: encoder/decoder
    # non-implemented/allt:encoder/decoder+transform/output+transform;
    # non-implemented/dec_outt: decoder/output+transform;
    # non-implemented/enc_dect: encoder/decoder+transform
    # none: no tied embedding
    tie_embedding = args.tied_embedding
    pretrained = args.pretrained
    if pretrained:
        pretrained_embedding = get_path('../text_simplification_data/glove/glove.840B.300d.txt', 'sys')
    else:
        pretrained_embedding = ''

    attention_type = args.attention_type

    data_base = 'data_plain'
    train_dataset_simple = get_path('data/%s/train_dummy_simple_dataset'%data_base, 'sys')
    train_dataset_simple_ext = get_path('data/%s/train_dummy_simple_dataset_ext' % data_base, 'sys')
    train_dataset_simple2 = get_path('data/%s/train_dummy_simple_dataset2'%data_base, 'sys')

    train_dataset_complex = get_path('data/%s/train_dummy_complex_dataset'%data_base, 'sys')
    train_dataset_complex2 = get_path('data/%s/train_dummy_complex_dataset2%data_base', 'sys')
    train_dataset_complex_ppdb = get_path('data/%s/train_dummy_complex_dataset.rules'%data_base, 'sys')
    train_dataset_complex_ppdb_cand = get_path('data/%s/train_dummy_complex_dataset_cand.rules'%data_base, 'sys')
    val_dataset_complex_ppdb = get_path('data/%s/eval_dummy_complex_dataset.rules'%data_base, 'sys')
    vocab_simple = get_path('data/%s/dummy_simple_vocab'%data_base, 'sys')
    vocab_complex = get_path('data/%s/dummy_complex_vocab'%data_base, 'sys')
    vocab_all = get_path('data/%s/dummy_vocab'%data_base, 'sys')
    vocab_rules = get_path('data/%s/dummy_rules_vocab'%data_base, 'sys')
    rule_mode = 'unigram'
    # if args.lower_case:
    #     vocab_simple = vocab_simple + '.lower'
    #     vocab_complex = vocab_complex + '.lower'
    #     vocab_all = vocab_all + '.lower'

    subword_vocab_size = args.subword_vocab_size

    if subword_vocab_size > 0:
        subword_vocab_simple = get_path('data/%s/dummy_subvocab'%data_base, 'sys')
        subword_vocab_complex = get_path('data/%s/dummy_subvocab'%data_base, 'sys')
        subword_vocab_all = get_path('data/%s/dummy_subvocab'%data_base, 'sys')
        max_complex_sentence = 100
        max_simple_sentence = 90

    val_dataset_simple_folder = get_path('data/%s/'%data_base, 'sys')
    val_dataset_simple_file = 'valid_dummy_simple_dataset'
    val_dataset_complex = get_path('data/%s/valid_dummy_complex_dataset'%data_base, 'sys')
    val_mapper = get_path('data/%s/valid_dummy_mapper'%data_base, 'sys')
    val_dataset_complex_rawlines_file = val_dataset_complex
    val_dataset_simple_rawlines_file_references = 'valid_dummy_simple_dataset.raw.'
    val_dataset_simple_rawlines_file = val_dataset_simple_file
    num_refs = 3

    output_folder = args.output_folder
    logdir = get_path('../' + output_folder + '/log/', environment)
    modeldir = get_path('../' + output_folder + '/model/', environment)
    resultdir = get_path('../' + output_folder + '/result/', environment)

    allow_growth = True
    # per_process_gpu_memory_fraction = 1.0
    use_mteval = True
    mteval_script = get_path('script/mteval-v13a.pl', 'sys')
    mteval_mul_script = get_path('script/multi-bleu.perl', 'sys')
    joshua_class = get_path('script/ppdb-simplification-release-joshua5.0/joshua/class', 'sys')
    joshua_script = get_path('script/ppdb-simplification-release-joshua5.0/joshua/bin/bleu', 'sys')
    corpus_sari_script = get_path('script/corpus_sari.sh', 'sys')
    corpus_sari_script_nonref = get_path('script/corpus_sari_nonref.sh', 'sys')

    path_ppdb_refine = get_path(args.path_ppdb_refine, 'sys')

    # For Exp
    penalty_alpha = args.penalty_alpha

    # For Memory
    memory = args.memory
    if memory is not None:
        memory = memory.split(':')
    else:
        memory = []
    max_cand_rules = 15
    memory_prepare_step = args.memory_prepare_step
    rule_threshold = args.rule_threshold
    memory_config = args.memory_config
    min_count_rule = 0
    if 'mincnt' in memory_config:
        # Assume single digit for min_count_rule
        cnt_idx = memory_config.index('mincnt') + len("mincnt")
        min_count_rule = float(memory_config[cnt_idx: cnt_idx+3])
    ctxly = None
    if 'ctxly' in memory_config:
        ctxly_idx = memory_config.index('ctxly') + len("ctxly")
        ctxly = int(memory_config[ctxly_idx: ctxly_idx+1])

    # For RL
    rl_config = args.rl_config
    # sari: add sari metric for optimize
    # sari_ppdb_simp_weight: the weight for ppdb in sari
    # sample: sari|rule|rule_weight:2.0
    rl_configs = {}
    rulecnt_threshold = 0
    for cfg in rl_config.split(':'):
        kv = cfg.split('=')
        if kv[0] == 'dummy':
            rl_configs['dummy'] = True
        if kv[0] == 'sari':
            rl_configs['sari'] = True
        if kv[0] == 'sari_weight':
            rl_configs['sari_weight'] = float(kv[1])
        if kv[0] == 'rule':
            rl_configs['rule'] = True
        if kv[0] == 'noneg':
            rl_configs['noneg'] = True
        if kv[0] == 'stopgrad':
            rl_configs['stopgrad'] = True

    rule_base = args.rule_base

    fetch_mode = None
    tune_mode = None
    tune_style = args.tune_style # PPDB_score:add_score:len_score:
    if tune_style is not None:
        tune_style = [float(v) for v in tune_style.split(':')]
    # assert len(tune_style) == 3


class DefaultTrainConfig(DefaultConfig):
    beam_search_size = 0


class DefaultTestConfig(DefaultConfig):
    environment = args.environment
    beam_search_size = 1
    batch_size = 2
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/test1', environment)


class WikiDressLargeNewDefault(DefaultConfig):
    batch_size = args.batch_size
    dimension = args.dimension
    min_count = args.min_count
    dmode = args.dmode
    model_print_freq = 100
    memory_prepare_step = args.memory_prepare_step

    train_dataset_complex = get_path('../text_simplification_data/train/wikilargenew/train/src.txt', 'sys')
    train_dataset_simple = get_path('../text_simplification_data/train/wikilargenew/train/dst.txt', 'sys')
    vocab_simple = get_path(
            '../text_simplification_data/train/wikilargenew/train/voc_dst.txt', 'sys')
    vocab_complex = get_path(
            '../text_simplification_data/train/wikilargenew/train/voc_src.txt', 'sys')
    vocab_all = get_path(
        '../text_simplification_data/train/wikilargenew/train/voc_all.txt', 'sys')
    vocab_rules = get_path('../text_simplification_data/train/wikilargenew/train/rule_voc.txt', 'sys')
    train_dataset_complex_ppdb = get_path('../text_simplification_data/train/wikilargenew/train/rule_mapper.txt', 'sys')
    if dmode == 'v2':
        train_dataset_complex = get_path('../text_simplification_data/train/wikilargenew/train/src2.txt', 'sys')
        train_dataset_simple = get_path('../text_simplification_data/train/wikilargenew/train/dst2.txt', 'sys')
        vocab_simple = get_path(
            '../text_simplification_data/train/wikilargenew/train/voc_dst2.txt', 'sys')
        vocab_complex = get_path(
            '../text_simplification_data/train/wikilargenew/train/voc_src2.txt', 'sys')
        vocab_all = get_path(
            '../text_simplification_data/train/wikilargenew/train/voc_all2.txt', 'sys')
        vocab_rules = get_path('../text_simplification_data/train/wikilargenew/train/rule_voc.txt', 'sys')
        train_dataset_complex_ppdb = get_path(
            '../text_simplification_data/train/wikilargenew/train/rule_mapper.txt', 'sys')

    val_dataset_simple_folder = get_path('../text_simplification_data/train/wikilargenew/val/', 'sys')
    val_dataset_simple_file = 'dst.txt'
    val_dataset_complex = get_path('../text_simplification_data/train/wikilargenew/val/src.txt', 'sys')
    val_mapper = get_path('../text_simplification_data/train/wikilargenew/val/map.txt', 'sys')

    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/train/wikilargenew/val/src.raw.txt', 'sys')
    val_dataset_simple_rawlines_file_references = 'ref.'
    val_dataset_simple_rawlines_file = 'dst.raw.txt'
    num_refs = 8

    max_complex_sentence = 115
    max_simple_sentence = 95


class WikiDressHugeNewDefault(WikiDressLargeNewDefault):
    train_dataset_complex = get_path('../text_simplification_data/train/wikihugenew/src.txt', 'sys')
    train_dataset_simple = get_path('../text_simplification_data/train/wikihugenew/dst.txt', 'sys')
    vocab_rules = get_path('../text_simplification_data/train/wikihugenew/rule_voc.txt', 'sys')
    train_dataset_complex_ppdb = get_path('../text_simplification_data/train/wikihugenew/rule_mapper.txt', 'sys')


class WikiDressHugeNewTrainDefault(WikiDressHugeNewDefault):
    beam_search_size = 0
    max_cand_rules = 15


class WikiDressLargeNewTrainDefault(WikiDressLargeNewDefault):
    beam_search_size = 0
    max_cand_rules = 15


class WikiDressLargeNewEvalDefault(WikiDressLargeNewDefault):
    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_val', environment)
    max_cand_rules = 50
    val_dataset_complex_ppdb = get_path('../text_simplification_data/train/wikilargenew/val/rule_mapper.txt', 'sys')


class WikiDressLargeNewEvalForBatchSize(WikiDressLargeNewDefault):
    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/''', environment)


class WikiDressLargeNewTestDefault(WikiDressLargeNewDefault):
    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_test', environment)

    val_dataset_simple_folder = get_path('../text_simplification_data/train/wikilargenew/test/', 'sys')
    val_dataset_simple_file = 'dst.txt'
    val_dataset_complex = get_path('../text_simplification_data/train/wikilargenew/test/src.txt', 'sys')
    val_mapper = get_path('../text_simplification_data/train/wikilargenew/test/map.txt', 'sys')

    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/train/wikilargenew/test/src.raw.txt', 'sys')
    val_dataset_simple_rawlines_file_references = 'ref.'
    val_dataset_simple_rawlines_file = 'dst.raw.txt'
    num_refs = 8

    max_cand_rules = 50
    val_dataset_complex_ppdb = get_path('../text_simplification_data/train/wikilargenew/test/rule_mapper.txt', 'sys')


########################################################################################## WIKILARGE


class WikiDressLargeDefault(DefaultConfig):
    model_print_freq = 100
    save_model_secs = 600
    model_eval_freq = args.model_eval_freq

    train_dataset_simple = get_path(
        '../text_simplification_data/train/wikilarge/wiki.full.aner.train.dst', 'sys')
    train_dataset_complex = get_path(
        '../text_simplification_data/train/wikilarge/wiki.full.aner.train.src', 'sys')
    vocab_rules = get_path(
        '../text_simplification_data/train/wikilarge/rule_voc.txt', 'sys')
    train_dataset_complex_ppdb = get_path(
        '../text_simplification_data/train/wikilarge/rule_mapper.txt', 'sys')
    dmode = args.dmode
    # if dmode == 'v2':
    #     train_dataset_simple = get_path(
    #         '../text_simplification_data/train/wikilarge/dst2.txt', 'sys')
    #     train_dataset_complex = get_path(
    #         '../text_simplification_data/train/wikilarge/src2.txt', 'sys')
    #     vocab_rules = get_path(
    #         '../text_simplification_data/train/wikilarge/rule_voc.txt', 'sys')
    #     train_dataset_complex_ppdb = get_path(
    #         '../text_simplification_data/train/wikilarge/rule_mapper.txt', 'sys')

    max_cand_rules = 15
    vocab_simple = get_path(
        '../text_simplification_data/train/wikilarge/wiki.full.aner.train.dst.vocab.lower', 'sys')
    vocab_complex = get_path(
        '../text_simplification_data/train/wikilarge/wiki.full.aner.train.src.vocab.lower', 'sys')
    # vocab_all = get_path(
    #     '../text_simplification_data/train/wikilarge/voc_all.txt', 'sys')

    val_dataset_simple_folder = get_path('../text_simplification_data/val/', 'sys')
    val_dataset_simple_file = 'tune.8turkers.tok.simp.ner'
    val_dataset_complex = get_path('../text_simplification_data/val/tune.8turkers.tok.norm.ner', 'sys')
    val_mapper = get_path('../text_simplification_data/val/tune.8turkers.tok.map', 'sys')
    # wiki.full.aner.ori.valid.dst is uppercase whereas tune.8turkers.tok.simp is lowercase
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/val/tune.8turkers.tok.norm', 'sys')
    val_dataset_simple_rawlines_file_references = 'tune.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'tune.8turkers.tok.simp'

    val_dataset_simple_raw_file = 'wiki.full.aner.ori.valid.dst'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/val/wiki.full.aner.ori.valid.src', 'sys')

    num_refs = 8

    dimension = args.dimension

    max_complex_sentence = 85
    max_simple_sentence = 85

    min_count = args.min_count
    batch_size = args.batch_size

    tokenizer = 'split'


class WikiDressLargeTrainDefault(WikiDressLargeDefault):
    beam_search_size = 0
    max_cand_rules = 15


class WikiDressLargeEvalDefault(WikiDressLargeDefault):
    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_val', environment)
    max_cand_rules = 50
    val_dataset_complex_ppdb = get_path('../text_simplification_data/val/rule_mapper.txt', 'sys')


class WikiDressLargeTestDefault(WikiDressLargeDefault):
    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_test', environment)

    val_dataset_simple_folder = get_path('../text_simplification_data/test/')
    # use the original dress
    val_dataset_simple_file = 'wiki.full.aner.test.dst'
    val_dataset_complex = get_path('../text_simplification_data/test/wiki.full.aner.test.src')
    val_mapper = get_path('../text_simplification_data/test/test.8turkers.tok.map.dress')
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/test/test.8turkers.tok.norm')
    val_dataset_simple_rawlines_file_references = 'test.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'test.8turkers.tok.simp'
    num_refs = 8

    max_cand_rules = 50
    val_dataset_complex_ppdb = get_path('../text_simplification_data/test/rule_mapper.txt', 'sys')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Config created for 2019
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class WikiTransDefaultConfig(DefaultConfig):
    fetch_mode = args.fetch_mode
    batch_size = args.batch_size
    dimension = args.dimension
    model_print_freq = 100
    save_model_secs = 600
    lower_case = args.lower_case

    subword_vocab_size = args.subword_vocab_size
    model_eval_freq = args.model_eval_freq
    tune_style = args.tune_style
    tune_mode = args.tune_mode
    if tune_mode is not None:
        tune_mode = tune_mode.split(':')
    if tune_style is not None:
        tune_style = [float(v) for v in tune_style.split(':')]

    # if subword_vocab_size == 30000:
    #     train_dataset = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdbxu/train.tfrecords.*'
    #     subword_vocab_complex = '/zfs1/hdaqing/saz31/dataset/vocab/comp30k.subvocab'
    #     subword_vocab_simple = '/zfs1/hdaqing/saz31/dataset/vocab/simp30k.subvocab'
    #     subword_vocab_all = ''
    #     max_complex_sentence = 180
    #     max_simple_sentence = 170
    # elif subword_vocab_size == 0:
    #     train_dataset = '/zfs1/hdaqing/saz31/dataset/trans_tf_example/ppdb_0_0k/train.tfrecords.*'
    #     vocab_complex = '/zfs1/hdaqing/saz31/dataset/vocab/comp.vocab'
    #     vocab_simple = '/zfs1/hdaqing/saz31/dataset/vocab/simp.vocab'
    #     subword_vocab_all = ''
    #     max_complex_sentence = 95
    #     max_simple_sentence = 85

    replace_unk_by_attn = False
    replace_unk_by_emb = True
    replace_unk_by_cnt = False
    dmode = args.dmode

    if fetch_mode == 'tf_example_dataset':
        print('Use Tf Example Dataset.')

        train_dataset = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdbxu/train.tfrecords.*'
        subword_vocab_complex = '/zfs1/hdaqing/saz31/dataset/vocab/comp30k.subvocab'
        subword_vocab_simple = '/zfs1/hdaqing/saz31/dataset/vocab/simp30k.subvocab'
        subword_vocab_all = ''
        max_complex_sentence = 180
        max_simple_sentence = 170

        if dmode == 'alter':
            if subword_vocab_size == 30000:
                train_dataset2 = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdbxu/wiki.tfrecords.*'
            else:
                raise NotImplementedError('')
        elif dmode == 'wk':
            if subword_vocab_size == 30000:
                train_dataset = '/zfs1/hdaqing/saz31/dataset/tf_example/ppdbxu/wiki.tfrecords..*'
            else:
                raise NotImplementedError('')
        else:
            train_dataset2 = None


class WikiTransTrainConfig(WikiTransDefaultConfig):
    beam_search_size = 1


class WikiTransEvalConfig(WikiTransDefaultConfig):
    fetch_mode = None
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_val', environment)

    beam_search_size = 1
    val_dataset_simple_folder = get_path('../text_simplification_data/val2/nsimp/', 'sys')
    val_dataset_simple_file = 'tune.8turkers.tok.simp'
    val_dataset_complex = get_path('../text_simplification_data/val2/ncomp/tune.8turkers.tok.norm', 'sys')
    val_dataset_complex_features = get_path('../text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.features', 'sys')
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori', 'sys')
    val_dataset_simple_rawlines_file_references = 'tune.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'tune.8turkers.tok.simp.ori'

    val_dataset_simple_raw_file = 'tune.8turkers.tok.simp.ori'
    val_dataset_complex_raw = get_path(
        '../text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori', 'sys')
    val_mapper = get_path('../text_simplification_data/val2/nmap/tune.8turkers.tok.map')
    num_refs = 8


class WikiTransTestConfig(WikiTransDefaultConfig):
    fetch_mode = None

    beam_search_size = 1
    environment = args.environment
    output_folder = args.output_folder
    resultdir = get_path('../' + output_folder + '/result/eightref_test', environment)

    val_dataset_simple_folder = get_path('../text_simplification_data/test2/nsimp/')
    val_dataset_simple_file = 'test.8turkers.tok.simp'
    val_dataset_complex = get_path('../text_simplification_data/test2/ncomp/test.8turkers.tok.norm')
    val_dataset_complex_features = get_path('../text_simplification_data/test2/ncomp/test.8turkers.tok.norm.features', 'sys')
    val_dataset_complex_rawlines_file = get_path(
        '../text_simplification_data/test2/ncomp/test.8turkers.tok.norm.ori')
    val_dataset_simple_rawlines_file_references = 'test.8turkers.tok.turk.'
    val_dataset_simple_rawlines_file = 'test.8turkers.tok.simp.ori'
    val_mapper = get_path('../text_simplification_data/test2/nmap/test.8turkers.tok.map')
    num_refs = 8


class WikiTransDummyConfig(WikiTransDefaultConfig):
    data_base = 'data_trans'
    subword_vocab_complex = get_path('data/%s/comp.subvocab' % data_base, 'sys')
    subword_vocab_simple = get_path('data/%s/simp.subvocab' % data_base, 'sys')
    train_dataset = get_path('data/%s/data.tfrecords' % data_base, 'sys')

    val_dataset_simple_folder = get_path('data/%s/' % data_base, 'sys')
    val_dataset_simple_file = 'valid_dummy_simple_dataset'
    val_dataset_complex = get_path('data/%s/valid_dummy_complex_dataset' % data_base, 'sys')
    val_mapper = get_path('data/%s/valid_dummy_mapper' % data_base, 'sys')
    val_dataset_complex_rawlines_file = val_dataset_complex
    val_dataset_simple_rawlines_file_references = 'valid_dummy_simple_dataset.raw.'
    val_dataset_simple_rawlines_file = val_dataset_simple_file
    num_refs = 3

    max_complex_sentence = 25
    max_simple_sentence = 20
    beam_search_size = 1
    save_model_secs = 10

def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output