#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:02:48 2019

@author: swinchurkar
"""

import numpy as np
np.set_printoptions(threshold=np.inf)


train_data_set ="data/train-trec.txt"
test_data_set ="data/test-trec.txt"

train_proc_data_set ="data/train-trec-proc.txt"
test_proc_data_set ="data/test-trec-proc.txt"

def process_data(data_set, proc_data_set):
    lines = 0
    texts = ""
    file = open(data_set, "r", encoding = 'utf-8')
    for line in file:
        words = line.split(":")
        length = len(line.split(" "))
        if words[0] == "ABBR":
            print("Skiping the class ABBR question ", line)
            lines += 1
        elif length >= 30:
            print("Exceding length {} skiping the question {}".format(length, line))
            lines += 1
        else:
            texts += line

    file.close()

    file = open(proc_data_set, "w", encoding = 'utf-8')
    file.write(str(texts))
    file.close()
    print("Total {} questions skiped".format(lines))


process_data(train_data_set, train_proc_data_set)
process_data(test_data_set, test_proc_data_set)

