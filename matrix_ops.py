# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:24:18 2019

@author: swinchurkar
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def single_instance(inp):
    b, a = inp
    return b  

A = [[[1,2,3,4], [1,2,3,4], [1,2,3,4]]]
     
B = [[2,2,3,2], [1,1,1,1], [4,4,4,4]]
W = [[2,2,2,2], [2,2,2,2], [2,2,2,2]]


a = K.variable(value=A)
b = K.variable(value=B)
w = K.variable(value=W)

print("shape of a: ", a)
print("shape of b: ", b)
print("shape of w: ", w)
print("a = \n", K.eval(a))

a = K.permute_dimensions(a, [0,2,1])
print("shape of a: ", a)
print("a = \n", K.eval(a))

#print("shape of b: ", b)
#print("b = \n", K.eval(b))

#z = K.dot(at, b)
#print("shape of z: ", z)
#print("z = \n", K.eval(z))
 



