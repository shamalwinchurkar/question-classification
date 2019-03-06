# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:56:37 2019

@author: swinchurkar
"""
import pickle
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.framework import dtypes

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class EmbeddingWeights(Initializer):
  def __init__(self, embedding_matrix=None, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)
    self.embedding_matrix = embedding_matrix
    
  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return self.embedding_matrix
 
  def get_config(self):
    return {
        "embedding_matrix": self.embedding_matrix,
        "dtype": self.dtype.name
    }
    
class Embeddings():
    def __init__(self, filename):
        print("Loading pre-trained embedding for QC data.")
        self.emb_dict = load_obj(filename)
        self.emb_dim = self.emb_dict["is"].shape[0]
        print("Dimensions of embedding is ", self.emb_dim)
        print("No of words in emb_dict ", len(self.emb_dict))
        
    def get_emb_matrix(self, vocabulary):
        voc_size = len(vocabulary)
        w2v = np.random.uniform(-0.25, 0.25, (voc_size, self.emb_dim))
        w2v = np.array(w2v, dtype = np.float32)
        word_not_found = 0
        for i in range(voc_size):
            word = vocabulary[i]
            if word is not None:
                try:
                    w2v[i] = self.emb_dict[word]
                except KeyError:
                    print("word {} at index {} not found in GN".format(word, i))
                    word_not_found +=1    
                
        print("No of words not found :", word_not_found)   
        return w2v
    
    def get_emb_dim(self):
        return self.emb_dim
        