# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:20:00 2019

@author: swinchurkar
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import qc_data
import qc_emb
import argparse
import os
import time
import qc_cnn
import qc_gru
import qc_c_gru
import qc_lstm
import qc_c_lstm
import qc_plot
import qc_a_cnn

from sklearn.metrics import confusion_matrix

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        
    def on_epoch_end(self, epoch, logs={}):
        print("\nTesting Test Data on epoch :", epoch)
        x, y, batch_size = self.test_data
        y_pred_cnn = self.model.predict(x, verbose=1, batch_size=batch_size)
        y_pred_cnn = (y_pred_cnn > 0.5)
        y_test_nc = [ np.argmax(t) for t in y ]
        y_pred_nc = [ np.argmax(t) for t in y_pred_cnn ]
        cm = confusion_matrix(y_test_nc, y_pred_nc)
        print("Confusion Matrix:")
        print(cm)
        length = len(cm[0])
        accuracy_cnn = 0
        for i in range(length):
            accuracy_cnn += cm[i][i]
        accuracy_cnn = (accuracy_cnn / len(y_test_nc)) * 100
        loss_cnn = 100 - accuracy_cnn
        print("\nTest Score: Accuracy {}".format(accuracy_cnn))
        print("\nTest Score: Loss {}".format(loss_cnn))
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(round(loss, 2),
              round(acc, 2)))

class SaveBestCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        self.best_val_acc = 0
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs={}):
        val_acc = round(logs['val_acc'], 4)
        if self.best_val_acc < val_acc:
            self.best_val_acc = val_acc
            print("\nSaved model weigths with val_acc {} to {}".format(
                    self.best_val_acc, self.filepath))
            self.model.save_weights(self.filepath)
        else:
            print("\nCurrent best val_acc is ", self.best_val_acc)
            
# Training Parameters
validation_samples = 500
dropout_rate = 0.5
       
def train_model(model_list, train_dataset, test_dataset, embedding_file, 
              atten_words_dict_file, batch_size, epochs):
    model_history = dict()
    checkpoint_history = dict()
    model_acc = []
    model_loss = []
        
    ds = qc_data.Dataset(train_dataset, test_dataset, atten_words_dict_file)
    x_train, y_train, x_val, y_val, x_test, y_test, x_train_atten, \
        x_val_atten, x_test_atten = ds.load(validation_samples)
         
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
    run_folder = 'run'
    model_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
  
    out_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder,
                                           timestamp))
    
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    logs_dir = os.path.abspath(os.path.join(out_dir, "logs"))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)    
        
    tboard_dir = os.path.join(out_dir, "tensorboard") 
    if not os.path.exists(tboard_dir):
            os.makedirs(tboard_dir)
            
    ds.print_stat(logs_dir)
    
    for model_name in model_list:
        checkpoint_file = os.path.join(checkpoint_dir,
                                           model_name +'-weights')
        
        tboard_m_dir = os.path.join(tboard_dir, model_name) 
        if not os.path.exists(tboard_m_dir):
            os.makedirs(tboard_m_dir)
        
        tboard = tf.keras.callbacks.TensorBoard(log_dir=tboard_m_dir, histogram_freq=1,
                            write_graph=True, write_images=False)
                
        if model_name == "cnn":
            # CNN
            model = qc_cnn.CNN(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['acc'])
            modelCheckpoint = SaveBestCallback(filepath = checkpoint_file)
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data=(x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=[modelCheckpoint])
            model = qc_cnn.CNN(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "bgru":
            model = qc_gru.BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9)
            model.compile(loss="categorical_crossentropy",
                          optimizer=RMSprop,
                          metrics=['acc'])
                                
            modelCheckpoint = SaveBestCallback(filepath = checkpoint_file)
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data=(x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=[modelCheckpoint])
            model = qc_gru.BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)
        elif model_name == "c-bgru":
            model = qc_c_gru.C_BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9)
            model.compile(loss="categorical_crossentropy",
                          optimizer=RMSprop,
                          metrics=['acc'])
                                
            modelCheckpoint = SaveBestCallback(filepath = checkpoint_file)
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data=(x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=[modelCheckpoint])
            model = qc_c_gru.C_BGRU(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)    
        elif model_name == "blstm":
            model = qc_lstm.BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9)
            model.compile(loss="categorical_crossentropy",
                          optimizer=RMSprop,
                          metrics=['acc'])
                                
            modelCheckpoint = SaveBestCallback(filepath = checkpoint_file)
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data=(x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=[modelCheckpoint])
            model = qc_lstm.BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)    
        elif model_name == "c-blstm":
            model = qc_c_lstm.C_BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9)
            model.compile(loss="categorical_crossentropy",
                          optimizer=RMSprop,
                          metrics=['acc'])
                                
            modelCheckpoint = SaveBestCallback(filepath = checkpoint_file)
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data=(x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=[modelCheckpoint])
            model = qc_c_lstm.C_BLSTM(emb_dim, voc_size, sen_len,
                              num_class, emb_matrix, 0)    
        elif model_name == "acnn":
            # ACNN
            model = qc_a_cnn.ACNN(emb_dim, voc_size, sen_len, atten_sen_len,
                              num_class, emb_matrix,
                              dropout_rate)
            model.summary()
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['acc'])
            
            history = model.fit(x_train, y_train, batch_size=batch_size, 
                                validation_data = (x_val, y_val), epochs=epochs, 
                                verbose=1, callbacks=None)
             
            #history = model.fit([x_train, x_train_atten], y_train, batch_size=batch_size, 
            #                    validation_data = ([x_val, x_val_atten], y_val), epochs=epochs, 
            #                    verbose=1, callbacks=None)
            
            return
        else:
            print("Requested model {} is not implemented".format(model_name))
            continue
        
        model.load_weights(checkpoint_file)            
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
        model_history[model_name] = history
        checkpoint_history[model_name] = checkpoint_file
        model_acc.append(accuracy)
        model_loss.append(loss)
        ds.write_predections(filename, y_test_nc, y_pred_nc)
    
    qc_plot.plot_graphs(logs_dir, model_list, model_history, model_acc,
                        model_loss, epochs)
      
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dataset", help="Dataset to be trained "
                                              "and evaluated.",
                      type=str, default="data/train-trec.txt")
  
    parser.add_argument("-r", "--test_dataset", help="Dataset to be tested",
                      type=str, default="data/test-trec.txt")

    parser.add_argument("-d", "--attention_words_dictionary", help="Attention words dictionary",
                      type=str, default="data/attention-words.txt")

    parser.add_argument("-e", "--embedding_file",
                      help="Embeding file to load for Embedding layer.",
                      type=str, default="data/gn_pre_trained_embeddings")
   
    parser.add_argument("-b", "--batch_size",
                      help="The size of each batch for training.",
                      type=int, default=50)

    parser.add_argument("-p", "--epochs",
                      help="The number of epochs for training.",
                      type=int, default=10)
    
    parser.add_argument("-m", "--model",
                      help="Model to train.",
                      type=str, 
                      choices = ["cnn", "bgru", "c-bgru", "blstm", "c-blstm",
                                 "mac-bgru", "acnn", "all"],
                      default="all")
    
    parser.add_argument("-n", "--mode", help="Train or test model",
                      type=str, choices = ["train", "test"], default="train")

    args = parser.parse_args()
    model_list = None
    if args.model == "all":
        model_list = ["cnn", "bgru", "c-bgru", "blstm", "c-blstm", "acnn"]
    else:
        model_list = [args.model]
        
    train_model(model_list, args.train_dataset, args.test_dataset, 
                args.embedding_file, args.attention_words_dictionary,
                args.batch_size,
                args.epochs)
    