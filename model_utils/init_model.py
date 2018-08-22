#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os
import time
import datetime
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from data_utils import utils
from conf.hyparam import Config
from model_utils import network
from utils.decoder.model import LM_decoder

tf.logging.set_verbosity(tf.logging.INFO)

class DeepSpeech2(object):
    ''' Class to init model

    :param wav_files: path to wav files
    :type wav_files: str
    :param text_labels: transcript for wav files
    :type text_labels: list
    :param words_size: the size of vocab
    :type words_size: int
    :param words : a list for vocab
    :type words: list
    :param word_num_map: a map dict from word to num
    :type word_num_map: dict
    return 
    '''
    def __init__(self, wav_files, text_labels, words_size, words, word_num_map):
        self.hyparam = Config()
        self.wav_files = wav_files
        self.text_labels = text_labels
        self.words_size = words_size
        self.words = words
        self.word_num_map = word_num_map
        # mfcc features contains 39 * 2 * 2 + 39 = 195 dims , linear specgram 161 dim which 161 = (8000-0)/50 + 1 
        self.n_dim = self.hyparam.n_input + 2 * self.hyparam.n_input * self.hyparam.n_context if self.hyparam.specgram_type == 'mfcc' else 161

    def add_placeholders(self):
        # input tensor for log filter or MFCC features
        self.input_tensor = tf.placeholder(tf.float32,
                                          [None, None, self.n_dim],
                                          name='input')
        self.text = tf.sparse_placeholder(tf.int32, name='text')
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
        self.keep_dropout = tf.placeholder(tf.float32)

    def deepspeech2(self):
        '''
        BUild a network with CNN-BRNN-Lookahead CNN -FC.
        '''
        batch_x = self.input_tensor
        seq_length = self.seq_length
        n_character = self.words_size + 1
        keep_dropout = self.keep_dropout
        n_input = self.hyparam.n_input
        n_context = self.hyparam.n_context

        batch_x_shape = tf.shape(batch_x)
        batch_x = tf.transpose(batch_x, [1, 0, 2])
        batch_x = tf.expand_dims(batch_x, -1)
        batch_x = tf.reshape(batch_x, 
                            [batch_x_shape[0], -1, self.n_dim, 1] ) # shape (batch_size, ?, n_dim, 1)

        with tf.variable_scope('conv_1'):
            # usage: conv2d(batch_x, filter_shape, strides, pool_size, hyparam, use_dropout=False)
            # Output is [batch_size, height, width. out_channel]
            conv_1 = network.conv2d(batch_x, 
                                    [1,  n_input / 3, 1,  1], # filter: [height, width, in_channel, out_channel]
                                    [1, 1, n_input/5 , 1], # strides: [1, height, width, 1]
                                    2, self.hyparam, use_dropout=False) # shape (8, ?, 19, 1)
            conv_1 = tf.squeeze(conv_1, [-1])
            conv_1 = tf.transpose(conv_1, [1, 0, 2])

        with tf.variable_scope('birnn_1'):
            birnn_1 = network.BiRNN(conv_1, seq_length, batch_x_shape, self.hyparam )
        with tf.variable_scope('birnn_2'):
            birnn_2 = network.BiRNN(birnn_1, seq_length, batch_x_shape, self.hyparam )
        with tf.variable_scope('birnn_3'):
            birnn_3 = network.BiRNN(birnn_2, seq_length, batch_x_shape, self.hyparam, use_dropout=True)
            birnn_3 = tf.reshape(birnn_3, [batch_x_shape[0], -1, 2*self.hyparam.n_cell_brnn])
            birnn_3 = tf.expand_dims(birnn_3, -1)
        with tf.variable_scope('lcnn_1'):
            # Lookahead CNN combines n time-steps in furture
#            lcnn_1 = network.lookahead_cnn(birnn_3, [2, 2*self.hyparam.n_cell_brnn, 1, 2*self.hyparam.n_cell_brnn], 2, seq_length, self.hyparam, use_dropout=True)
            lcnn_1 = network.conv2d(birnn_3,
                                    [1, n_input, 1, 1],
                                    [1, 1, n_input/5, 1],
                                    2, self.hyparam, use_dropout=True)
            lcnn_1 = tf.squeeze(lcnn_1,[-1])
            width_lcnn1 = 14 # use to compute lcnn_1[-1], computed by pool_size * n_input / 5
            lcnn_1 = tf.reshape(lcnn_1, [-1, int(math.ceil(2*self.hyparam.n_cell_brnn/width_lcnn1))])

        with tf.variable_scope('fc'):
            b_fc = self.variable_on_device('b_fc', [n_character], tf.random_normal_initializer(stddev=self.hyparam.b_stddev))
            h_fc = self.variable_on_device('h_fc', 
                                           [int(math.ceil(2*self.hyparam.n_cell_brnn/width_lcnn1)), n_character],
                                           tf.random_normal_initializer(stddev=self.hyparam.h_stddev))
            layer_fc = tf.add(tf.matmul(lcnn_1, h_fc), b_fc)
            # turn it to 3 dim, [n_steps, hyparam.batch_size, n_character]
            layer_fc = tf.reshape(layer_fc, [-1, batch_x_shape[0], n_character])

        self.logits = layer_fc

    def loss(self):      
        """ Define loss
        return
        """              
        # ctc loss   
        with tf.name_scope('loss'):
            self.avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(self.text, self.logits, self.seq_length))
            tf.summary.scalar('loss',self.avg_loss)
        # [optimizer]    
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hyparam.learning_rate).minimize(self.avg_loss)
                         
        with tf.name_scope("decode"):
            self.decoded, log_prob = ctc_ops.ctc_beam_search_decoder(self.logits, self.seq_length, merge_repeated=False)

        with tf.name_scope("ctc_beam_search_decode"):
            self.prob = tf.nn.softmax(self.logits, dim=0)
            self.prob = tf.transpose(self.prob, [1, 0, 2]) # keep the same dim with decoder {batch_size, time_step, n_character}
            self.decoder = LM_decoder(self.hyparam.alpha, self.hyparam.beta, self.hyparam.lang_model_path, self.words)

        with tf.name_scope("accuracy"):
            self.distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.text)
            # compute label error rate (accuracy)
            self.label_err = tf.reduce_mean(self.distance, name='label_error_rate')
            tf.summary.scalar('accuracy', self.label_err)


    def get_feed_dict(self, dropout=None):
        """ Define feed dict

        :param dropout: 
        :return:    
        """         
        feed_dict = {self.input_tensor: self.audio_features,
                     self.text: self.sparse_labels,
                     self.seq_length: self.audio_features_len}
                    
        if dropout != None:
            feed_dict[self.keep_dropout] = dropout
        else:       
            feed_dict[self.keep_dropout] = self.hyparam.keep_dropout_rate
                    
        return feed_dict
                    
    def init_session(self):
        self.savedir = self.hyparam.savedir
        self.saver = tf.train.Saver(max_to_keep=1)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # if no models , init it
        self.sess.run(tf.global_variables_initializer())
                    
        ckpt = tf.train.latest_checkpoint(self.savedir)
        tf.logging.info("Latest checkpoint: %s", (ckpt))
        self.startepo = 0
        if ckpt != None:
            self.saver.restore(self.sess, ckpt)
            ind = ckpt.rfind("-")
            self.startepo = int(ckpt[ind + 1:])
            tf.logging.info("Start epoch: %d", (self.startepo + 1))
                          
    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.hyparam.tensorboardfile, self.sess.graph)

    def init_decode(self):
        self.prob = tf.nn.softmax(self.logits, dim=0)
        self.prob = tf.transpose(self.prob, [1, 0, 2]) # keep the same dim with decoder {batch_size, time_step, n_character}
        self.decoder = LM_decoder(self.hyparam.alpha, self.hyparam.beta, self.hyparam.lang_model_path, self.words)

    def lm_decode(self, probality):
        result_transcripts = self.decoder.decode_batch_beam_search(
                probs_split=probality,
                beam_alpha=self.hyparam.alpha,
                beam_beta=self.hyparam.beta,
                beam_size=self.hyparam.beam_size,
                cutoff_prob=self.hyparam.cutoff_prob,
                cutoff_top_n=self.hyparam.cutoff_top_n,
                vocab_list=self.words,
                num_processes=self.hyparam.num_proc_bsearch)
        results = "" if result_transcripts == None else result_transcripts
        return  results[0].encode('utf-8')
                          
    def train(self):      
        epochs = self.hyparam.num_epoch
        batch_step_interval = self.hyparam.batch_step_interval
        tf.logging.info("Start training...")

        train_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            if epoch < self.startepo:
                continue

            tf.logging.info("Current epoch: %d/%d %s" % (epoch + 1, epochs, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            n_batches_epoch = int(np.ceil(len(self.text_labels) / self.hyparam.batch_size))
            tf.logging.info("Batch size: %d" % (self.hyparam.batch_size))

            train_cost = 0
            train_err = 0
            next_idx = 0  

            for batch in range(n_batches_epoch):
                tf.logging.info("Current batch: %d/%d %s" % (batch + 1, n_batches_epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                next_idx, self.audio_features, self.audio_features_len, self.sparse_labels, wav_files = utils.next_batch(
                    next_idx,
                    self.hyparam.batch_size,
                    self.hyparam.n_input,
                    self.hyparam.n_context,
                    self.text_labels,
                    self.wav_files,
                    self.word_num_map,
                    specgram_type=self.hyparam.specgram_type)
 
                batch_cost, _ = self.sess.run([self.avg_loss, self.optimizer], feed_dict=self.get_feed_dict())
                train_cost += batch_cost
 
                if (batch + 1) % batch_step_interval == 0:
                    tf.logging.info('Start merge: %s', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    rs = self.sess.run(self.merged, feed_dict=self.get_feed_dict())
                    self.writer.add_summary(rs, batch)

                    tf.logging.info("Current batch: %d/%d, loss: %.3f, error rate: %.3f" % (
                        batch + 1, n_batches_epoch, train_cost / (batch + 1), train_err))
                    tf.logging.info('Start decode: %s', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    d, train_err = self.sess.run([self.decoded[0], self.label_err], feed_dict=self.get_feed_dict(dropout=1.0))
                    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=self.sess)
                    dense_labels = utils.trans_tuple_to_texts_ch(self.sparse_labels, self.words)

                    for orig, decoded_array in zip(dense_labels, dense_decoded):
                        # convert to strings
                        decoded_str = utils.trans_array_to_text_ch(decoded_array, self.words)
                        tf.logging.info("Reference: %s" % (orig))
                        tf.logging.info("Transcript: %s" % (decoded_str))
                        break
 
            epoch_duration = time.time() - epoch_start
            tf.logging.info("Epoch: %d/%d, loss: %.3f, error rate: %.3f, time: %.2f sec" % (
                epoch + 1, epochs, train_cost, train_err, epoch_duration))
            self.saver.save(self.sess, os.path.join(self.savedir, self.hyparam.savefile), global_step=epoch)
                                   
        train_duration = time.time() - train_start
        tf.logging.info("Training complete, total duration: %2.f min" % (train_duration / 60))
        self.sess.close()
                                   
    def test(self):                
        index = 0                  
        next_idx = 20              
                                   
        for index in range(10):    
            next_idx, self.audio_features, self.audio_features_len, self.sparse_labels, wav_files = utils.next_batch(
               next_idx,           
               1,                  
               self.hyparam.n_input,            
               self.hyparam.n_context,          
               self.text_labels,   
               self.wav_files,     
               self.word_num_map,
               specgram_type=self.hyparam.specgram_type) 
                                   
            print 'Load wav file: ', wav_files[0]
            print 'Recognizing......'
                                   
            prob, d, train_ler = self.sess.run([self.prob, self.decoded[0], self.label_err], feed_dict=self.get_feed_dict(dropout=1.0))
            dense_labels = utils.trans_tuple_to_texts_ch(self.sparse_labels, self.words)
            
            if self.hyparam.use_lm_decoder:
                result_transcripts = self.lm_decode(prob)
                print "Orinal text: ", dense_labels[0]
                print "Transcript: ", result_transcripts
            else:
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=self.sess)
                print "dense_decoded", np.shape(dense_decoded), dense_decoded
                for orig, decoded_array in zip(dense_labels, dense_decoded):
                # turn to string        
                    decoded_str = utils.trans_array_to_text_ch(decoded_array, self.words)
                    print "Orinal text:", orig
                    print "Transcript: ", decoded_str
                    break               
#        self.sess.close()
         
    def recon_wav_file(self, wav_files, txt_labels):
        self.audio_features, self.audio_features_len, text_vector, text_vector_len = utils.get_audio_mfcc_features(
            None,
            wav_files,
            self.hyparam.n_input,
            self.hyparam.n_context,
            self.word_num_map,
            txt_labels,
            specgram_type=self.hyparam.specgram_type)
        self.sparse_labels = utils.sparse_tuple_from(text_vector)
        prob, d, train_ler = self.sess.run([self.prob, self.decoded[0], self.label_err], feed_dict=self.get_feed_dict(dropout=1.0))
        if self.hyparam.use_lm_decoder:
            result_transcripts = self.lm_decode(prob)
        else:
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=self.sess)
            result_transcripts = utils.trans_array_to_text_ch(dense_decoded[0], self.words).encode('utf-8')
#        print "Transcript: ", result_transcripts
        return result_transcripts
         
#        self.sess.close()
         
    def build_train(self):
        self.add_placeholders()
        self.deepspeech2()
        self.loss()
        self.init_session()
        self.add_summary()
        self.train()

    def build_test(self):
        self.add_placeholders()    
        self.deepspeech2()
        self.loss() 
        self.init_session()
        self.init_decode()
        self.test() 
                    
    def init_online_model(self):
        self.add_placeholders()
        self.deepspeech2()
        self.loss() 
        self.init_session()

    def predict(self, wav_files, txt_labels):
        transcript = self.recon_wav_file(wav_files, txt_labels)
        return transcript
    
    def close_onlie(self):
        self.sess.close()
                    
    def variable_on_device(self, name, shape, initializer):
#        with tf.device('/gpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        return var  

