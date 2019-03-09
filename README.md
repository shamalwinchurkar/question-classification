# Question Classification using Deep Neural Netowrk Models

# Introduction

This project attempts to solve question type classification problem using Deep Learning methods.I have implemented various
Question Classification models in Tensorflow Keras APIs 2.1.6-tf.


# Data Source

Dataset has following 6 distinct question types and 50 sub-types and it is divided into 2 files train and test.

|Primary Classes| ENTITY, NUMERIC, LOCATION, HUMAN, DESCRIPTION, ABBREVIATION|
|---------------|------------------------------------------------------------|

Training data set has total 5452 samples questions. Training data is divided into 4952 training samples and 500 validation samples. Test data set has total 500 sample questions.
 
TREC question type dataset is avaiable at follwing link [http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label]

# Related Work

## A C-LSTM neural network for text classification by chunting zhou1, chonglin sun2, zhiyuan liu3, francis C.M. Lau1
This paper presented the combined convolution and LSTM network for text classification.
https://arxiv.Org/abs/1511.08630

## ABCNN: attention-based convolutional neural network for modeling sentence pairs by wenpeng yin, hinrich sch¨ utze, bing xiang, bowen zhou
This paper presented the introduction of attention mechanism in convolution neural network for text classification.
https://arxiv.Org/abs/1512.05193

## Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification by Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
http://anthology.aclweb.org/P16-2034

## Other related works
Relation classification via tca cnns 
https://books.google.co.in/books?id=H2c9DwAAQBAJ&pg=PA141&lpg=PA141&dq=how+to+construct+diagonal+attention+matrix&source=bl&ots=FWqMF2TInF&sig=ACfU3U33PGqiXBG4Fcjc1Ry57Gd0AiQd2g&hl=en&sa=X&ved=2ahUKEwizrprm39rgAhWC458KHUhMA9kQ6AEwDHoECAAQAQ#v=onepage&q=how%20to%20construct%20diagonal%20attention%20matrix&f=false

## Code reused from other Projects
Some of the following functions
clean_text, pad_sentences, build_vocab, batch_iter are taken from follwing
projects
https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
https://github.com/thtrieu/qclass_dl/blob/master/data_helpers.py

# How to run?
Program can run in 2 modes, Training and Testing mode and Testing mode. In training mode, it runs the traning alogrithm on seleted model or all models with give or default training data set and then test the model with given or default test data set.

python qc_train_test.py [command line options]

Following are the command line options:

|Options |Description |Default Value |
|--------|------------|--------|
|-t, --train_dataset |Dataset to be trained and evaluated.|data/train-trec.txt| 
|-r, --test_dataset |Dataset to be tested| data/test-trec.txt|
|-d, --attention_words_dictionary |Attention words dictionary |data/attention-words.txt|
|-e, --embedding_file |Embedding file to load for Embedding layer |data/gn_pre_trained_embeddings|
|-b, --batch_size| The size of each batch for training| 50|
|-p, --epochs |The number of epochs for training |30|
|-l, --validation_samples |The number of samples for validation |500|    
|-c, --dropout_rate |Dropout Rate |0.20|
|-m, --model |Model to train. Choices are cnn, bgru, c-bgru, blstm, c-blstm, a-cnn, a-blstm, a-bgru, c-a-blstm, c-a-bgru, mac-blstm, mac-bgru, all |all
|-n, --mode |Train or test model. Choices are train, test |train|


