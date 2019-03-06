# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 04:22:32 2019

@author: swinchurkar
"""
import tensorflow as tf
import qc_emb

'''
class C_BLSTM(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length,
               class_dim, embedding_matrix, dropout_rate):
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_layer)
        conv1D_layer_3 = tf.keras.layers.Conv1D(emb_dim, 3, activation="relu")(layer)
        conv1D_layer_3 = tf.keras.layers.MaxPool1D(3, 1)(conv1D_layer_3)
        conv1D_layer_4 = tf.keras.layers.Conv1D(emb_dim, 4, activation="relu")(layer)
        conv1D_layer_4 = tf.keras.layers.MaxPool1D(4, 1)(conv1D_layer_4)
        conv1D_layer_5 = tf.keras.layers.Conv1D(emb_dim, 5, activation="relu")(layer)
        conv1D_layer_5 = tf.keras.layers.MaxPool1D(5, 1)(conv1D_layer_5)
        layer = tf.keras.layers.concatenate([conv1D_layer_3, conv1D_layer_4, conv1D_layer_5], axis=1)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(emb_dim * 2))(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        output = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer)
        super(C_BLSTM, self).__init__(inputs=[input_layer], outputs=output)
'''
        
class C_BLSTM(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length,
               class_dim, embedding_matrix, dropout_rate):
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_layer)
        layer = tf.keras.layers.Conv1D(emb_dim, 3, activation="relu")(layer)
        layer = tf.keras.layers.MaxPool1D(3, 1)(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(emb_dim * 2))(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        output = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer)
        super(C_BLSTM, self).__init__(inputs=[input_layer], outputs=output)