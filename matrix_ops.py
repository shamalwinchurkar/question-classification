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

A = [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
B = [[2,2,3,2], [1,1,1,1], [4,4,4,4]]
W = [[2,2,2,2], [2,2,2,2], [2,2,2,2]]

a = K.variable(value=A)
b = K.variable(value=B)
w = K.variable(value=W)

print("shape of a: ", a)
print("shape of b: ", b)
print("shape of w: ", w)

v = w * a

z = K.map_fn(single_instance, )

print("shape of z: ", z)
print("z = \n", K.eval(z))
 
#print("shape of z: ", z)
#print("shape of v: ", v)
#print("z = \n", K.eval(z))
#print("sum = \n", K.eval(sum))


#I = K.eye(3, dtype=tf.int32)
#print("shape of I: ", I)
#print("I = \n", K.eval(I))

#I = K.expand_dims(I, -1)
#print("shape of I: ", I)
#print("I = \n", K.eval(I))

#I = K.expand_dims(I, -1)
#print("shape of I: ", I)
#print("I = \n", K.eval(I))

#print("shape of x: ", x)
#print("x = ", K.eval(x))

#print("shape of y: ", y)
#print("y = ", K.eval(y))

#y = K.transpose(y)
#print("y = \n", K.eval(y))
#z = K.dot(x,y)
#z = K.transpose(z)
# Here you need to use K.eval() instead of z.eval() because this uses the backend session

#print("z = ", K.eval(z))
