# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 04:22:32 2019

@author: swinchurkar
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
import qc_emb

class Attention(tf.keras.layers.Layer):
    def __init__(self, 
                 use_bias=True,
                 weight_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 weight_regularizer=None,
                 bias_regularizer=None,
                 weight_constraint=None,
                 bias_constraint=None,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.use_bias = use_bias
        self.weight_initializer = initializers.get(weight_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.weight_regularizer = regularizers.get(weight_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.weight_constraint = constraints.get(weight_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise Exception('Input to Attention must be a list of '
                            'two tensors [sentence_input, attn_input].')     
        input_shape, atten_input_shape = input_shape
        batch, words, dim = atten_input_shape
                        
        self.atten_weight = self.add_weight(
        name='atten_weight',
        shape=(int(words), int(dim)),
        initializer=self.weight_initializer,
        regularizer=self.weight_regularizer,
        constraint=self.weight_constraint,
        trainable=True,
        dtype=tf.float32)
        if self.use_bias:
            self.atten_bias = self.add_weight(
            name='atten-bias',
            shape=(int(dim),),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        else:
            self.atten_bias = None
            
        super().build(input_shape)
        self.built = True
        
        print("atten_weight = ", self.atten_weight)
        print("bias = ", self.atten_bias)
        
    def call(self, input):
        input, atten_input = input
        print("call: input: ", input)
        w = self.atten_weight * atten_input[0] 
        print("call: w: ", w)
                
        
        output = input
        return output
    
    def compute_output_shape(self, input_shape):
        input_shape, atten_input_shape = input_shape
        print("compute_output_shape : input_shape : ", input_shape)
        return input_shape

class MyLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[1]), self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        print("build: shape of input: ", input_shape)
        print("build: shape of kernel: ", self.kernel)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        print("call: dot of x & kernel: ", K.dot(x, self.kernel))
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
'''
class ACNN(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        input = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input)
                
        layer = tf.keras.layers.Conv1D(emb_dim, 3, activation="relu")(layer)
        layer = tf.keras.layers.MaxPool1D(3, 1)(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Flatten()(layer)
        output = MyLayer(class_dim)(layer)
        
        super(ACNN, self).__init__(inputs=input, outputs=output)
'''

class ACNN(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        input_s = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer_s = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_s)
        
        input_p = tf.keras.layers.Input(shape=(atten_sen_len,), dtype=tf.int32)
        layer_p = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_p)
        layer = Attention()([layer_s, layer_p])
        layer = tf.keras.layers.Conv1D(emb_dim, 3, activation="relu")(layer)
        layer = tf.keras.layers.MaxPool1D(3, 1)(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        layer = tf.keras.layers.Flatten()(layer)
        output = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer) 
        super(ACNN, self).__init__(inputs=[input_s, input_p], outputs=[output])


'''        
class ACNN(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        input_s = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        output_s = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_s)
        
        layer = tf.keras.layers.Flatten()(output_s)
        output_s = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer) 
        
        input_p = tf.keras.layers.Input(shape=(atten_sen_len,), dtype=tf.int32)
        output_p = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_p)
        
        layer = tf.keras.layers.Flatten()(output_p)
        output_p = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer) 
        
        super(ACNN, self).__init__(inputs=[input_s, input_p], outputs=[output_s, output_p])
'''