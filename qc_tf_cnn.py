# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 04:22:32 2019

@author: swinchurkar
"""
import tensorflow as tf
import qc_emb

class ACNN(object):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        self.embedding_mat = embedding_matrix
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_dim], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_reg_lambda=0.000
        l2_loss = tf.constant(0.0001)

        # Extend input to a 4D Tensor, because tf.nn.conv2d requires so.
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(self.embedding_mat, name = "W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        
        filter_size = 3
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, emb_dim, 1, emb_dim]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[emb_dim]), name="b")
            conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            self.h_pool = tf.nn.max_pool(
                    h,
                    ksize=[1, sentence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
        
        # Combine all the pooled features
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, emb_dim])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.W = tf.Variable(tf.truncated_normal([emb_dim, class_dim],
                                                     stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[class_dim]), name="b")
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,
                                                                labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                           name="accuracy")
        # Predections
        with tf.name_scope("pred"):
            self.pred = self.predictions