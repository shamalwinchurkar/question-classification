# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 04:22:32 2019

@author: swinchurkar
"""
import tensorflow as tf
import qc_emb
import tensorflow.keras.backend as K
        
class C_A_BGRU(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length,
               class_dim, embedding_matrix, dropout_rate=0.2):
        units = emb_dim
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        embedding = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim,
                                          mask_zero=False)(input_layer)
            
        layer = tf.keras.layers.Conv1D(units, 3, activation="relu")(embedding)
        layer = tf.keras.layers.MaxPool1D(3, 1)(layer)
        activations = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units,
                                                return_sequences=True,
                                                dropout=dropout_rate
                                                ))(layer)
        attention = tf.keras.layers.Dense(1, activation="tanh")(activations)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(units * 2)(attention)
        attention = tf.keras.layers.Permute([2,1])(attention)
        qc = tf.keras.layers.Multiply()([activations, attention])
        qc = tf.keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2),
                                    output_shape=(units * 2,))(qc)
                
        output = tf.keras.layers.Dense(class_dim, activation="softmax", 
                kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(qc)
        super(C_A_BGRU, self).__init__(inputs=[input_layer], outputs=output)