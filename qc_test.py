# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:20:00 2019

@author: swinchurkar
"""
import numpy as np
import tensorflow as tf
import qc_data
import qc_emb
import os
import time
import qc_cnn
import qc_gru
import qc_a_gru
import qc_c_gru
import qc_c_a_gru
import qc_lstm
import qc_a_lstm
import qc_c_lstm
import qc_c_a_lstm
import qc_plot
import qc_utils

from sklearn.metrics import confusion_matrix
            
def test_model(trained_models_file, train_dataset, test_dataset, 
                embedding_file, atten_words_dict_file,
                batch_size):
    model_acc = []
    model_loss = []
    model_list = []
    model_dict = qc_utils.read_model_cfg(trained_models_file)
    ds = qc_data.Dataset(train_dataset, test_dataset, atten_words_dict_file)
    x_train, y_train, x_val, y_val, x_test, y_test, x_train_atten, \
        x_val_atten, x_test_atten = ds.load(0)
         
    emb = qc_emb.Embeddings(embedding_file)
    emb_matrix = emb.get_emb_matrix(ds.vocabulary_inv)
    emb_dim = emb.get_emb_dim()
    voc_size = ds.get_voc_size()
    num_class = ds.get_num_class()
    sen_len = x_train.shape[1]
    atten_sen_len = x_train_atten.shape[1]
    
    print("atten_sen_len = ", atten_sen_len)  
    print("Tensorflow Keras Version ", tf.keras.__version__)
    print("Embedding Dimensions ", emb_dim)
    print("Train/Validation/Test split: {:d}/{:d}/{:d}".format(len(y_train),
          len(y_val), len(y_test)))
    
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    run_folder = 'run/test'
    model_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
  
    out_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder,
                                           timestamp))
               
    logs_dir = os.path.abspath(os.path.join(out_dir, "logs"))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)    
                       
    ds.print_stat(logs_dir)
    
    for model_name, model_file in model_dict.items():
        print("model name {} model_file {}".format(model_name, model_file))
        if model_name == "cnn":
            print("CNN Model choosen")
            model = qc_cnn.CNN(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "bgru":
            model = qc_gru.BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        
        elif model_name == "c-bgru":     
            model = qc_c_gru.C_BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "a-bgru":
            model = qc_a_gru.A_BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "c-a-bgru":
            model = qc_c_a_gru.C_A_BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "blstm":
            model = qc_lstm.BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "a-blstm":
            model = qc_a_lstm.A_BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)    
        elif model_name == "c-blstm":
            model = qc_c_lstm.C_BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)    
        elif model_name == "c-a-blstm":
            model = qc_c_a_lstm.C_A_BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        else:
            print("Requested model {} is not implemented".format(model_name))
            continue
    
        model.load_weights(model_file)
        y_pred_cnn = model.predict(x_test, verbose=1, batch_size=batch_size)
        y_pred_cnn = (y_pred_cnn > 0.5)
        y_test_nc = [ np.argmax(t) for t in y_test ]
        y_pred_nc = [ np.argmax(t) for t in y_pred_cnn ]
        cm = confusion_matrix(y_test_nc, y_pred_nc)
        print("{} Confusion Matrix:".format(model_name.upper()))
        print(cm)
        
        filename = os.path.join(logs_dir, model_name + "-test-predictions.csv")
        length = len(cm[0])
        accuracy = 0    
        for i in range(length):
            accuracy += cm[i][i]
        accuracy = round((accuracy / len(y_test_nc)), 4)
        loss = round(1 - accuracy, 2)
        print("\n{} CM Test Score: Accuracy {}".format(model_name.upper(), 
              accuracy))
        print("{} CM Test Score: Loss {}".format(model_name.upper(), loss))
        model_list.append(model_name.upper())        
        model_acc.append(accuracy)
        model_loss.append(loss)
        ds.write_predections(filename, y_test_nc, y_pred_nc)    
    
    filename = os.path.join(logs_dir, 'test-results.csv')
    qc_plot.write_test_results(filename, model_list, model_acc, model_loss)
    
    return