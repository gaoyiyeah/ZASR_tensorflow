#!/usr/bin/env python
# coding=utf-8


class Config():
    '''
    Class to save all config in traing and decoding
    :param: None
    '''
    def __init__(self):

        self.batch_size = 8

        # Network hyparam
        self.n_brnn_layers = 3
        self.n_cell_brnn = 512
        self.learning_rate = 0.001
        self.keep_dropout_rate = 0.95
        self.b_stddev = 0.046875
        self.h_stddev = 0.046875

        # Feature
        self.n_input = 39  # 计算MFCC的个数
        self.n_context = 2  # n gram around current frame
        self.specgram_type = 'linear' # if 'linear' use specgram. 'mfcc' use mfcc with mfcc + delta1 + delta2
        self.use_bn = True # Batch normalization

        # Decoder
        self.use_lm_decoder = True # Whether use lm decoder. If False, use tf.ctc_beam_search_decoder
        self.alpha = 1.2
        self.beta = 2.5
        self.cutoff_prob = 0.99
        self.cutoff_top_n = 10
        self.num_proc_bsearch = 8
        self.beam_size = 400
        self.lang_model_path = 'models/lm/zh_giga.no_cna_cmn.prune01244.klm' # you can download it in https://github.com/PaddlePaddle/DeepSpeech

        # Config path
        self.vocab_path = u'data/aishell/vocab.txt'
        self.wav_path = u'/mnt/client/lustre_cpu/gaoyi01/aishell/data_aishell/wav/'
        self.label_file = u'/mnt/client/lustre_cpu/gaoyi01/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
        self.savedir = u'/mnt/client/lustre_cpu/gaoyi01/aishell/data_aishell'
        self.savefile = u'speech.cpkt'
        self.tensorboardfile = u'/mnt/client/lustre_cpu/gaoyi01/aishell/data_aishell/log'

        self.batch_step_interval = 15000
        self.num_epoch = 120