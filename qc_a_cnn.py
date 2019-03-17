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
        i_batch, i_words, i_dim = input_shape
        self.i_words = int(i_words)
        self.i_dim = int(i_dim)
        self.a_words = int(words)
        self.a_dim = int(dim)       
        #self.input_shape = input_shape
        #self.atten_shape = atten_input_shape
        
        super().build(input_shape)
        
    def call(self, input):
        input, atten_input = input
            
        print("call: atten_input: ", atten_input)
        print("call: input: ", input)
        
        # Prepare Attention vector
        
        output = tf.keras.layers.Dense(self.a_dim)(atten_input)
        print("call: output: ", output)
        output = tf.keras.layers.RepeatVector(self.i_words)(output)
        print("call: repeat output: ", output)
                        
        # Prepare Input vecotr
        '''
        output = input
        
        input_tensors = tf.split(input, self.i_words, axis=1)
        print("call: input_tensors: ", input_tensors)
        for i in input_tensors:
            i = K.repeat_elements(i, self.a_words, 1)
            print("call: i: ", i)
            print("call: w: ", w)
            m = tf.keras.layers.Multiply()([i, w])
            print("call: m: ", m)
            sum = tf.keras.layers.Lambda(lambda xin: K.sum(xin, axis=1))(m)
            print("call: sum: ", sum)
            sum = K.expand_dims(sum, axis=1)
            print("call: sum: ", sum)
            output.append(sum)
        
        output = tf.keras.layers.concatenate(output, axis=1)
        '''
        
        #output = tf.keras.layers.Activation("softmax")(output)
        #output = tf.keras.layers.Multiply()([output, input])
        #print("call: output: ", output)
        #o_shape=(1, int(self.i_words), int(self.i_dim))
        
        #output = K.reshape(output, o_shape)
        #output = tf.keras.layers.Reshape(o_shape)(output)
        #print("call: output: ", output)
        return input
    
    def compute_output_shape(self, input_shape):
        input_shape, atten_input_shape = input_shape
        print("compute_output_shape : atten_input_shape : ", atten_input_shape)
        return input_shape


class ACNN(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        input_s = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        activations = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_s)
                       
        input_p = tf.keras.layers.Input(shape=(atten_sen_len,), dtype=tf.int32)
        attention = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_p)
        layer = attention = Attention()([activations, attention])        
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
'''
class ACNN(tf.keras.models.Model):
    def __init__(self, emb_dim, num_words, sentence_length, atten_sen_len,
               class_dim, embedding_matrix, dropout_rate):
        
        input_s = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        activations = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_s)
        
                
        input_p = tf.keras.layers.Input(shape=(atten_sen_len,), dtype=tf.int32)
        attention = tf.keras.layers.Embedding(num_words,
                                          embeddings_initializer=
                                          qc_emb.EmbeddingWeights(embedding_matrix), 
                                          output_dim=emb_dim)(input_p)
        attention = tf.keras.layers.Dense(1, activation="tanh")(attention)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(emb_dim)(attention)
        attention = tf.keras.layers.Permute([2,1])(attention)
        qc = tf.keras.layers.Multiply()([activations, attention])
        #layer = tf.keras.layers.Conv1D(emb_dim, 3, activation="relu")(layer)
        #layer = tf.keras.layers.MaxPool1D(3, 1)(layer)
        #layer = tf.keras.layers.Dropout(dropout_rate)(layer)
        #layer = tf.keras.layers.Flatten()(layer)
        output = tf.keras.layers.Dense(class_dim, activation="softmax", 
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.0) )(layer) 
        super(ACNN, self).__init__(inputs=[input_s, input_p], outputs=[output])

'''
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