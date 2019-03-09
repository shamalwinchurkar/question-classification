# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:20:00 2019

@author: swinchurkar
"""
import argparse
import qc_train
import qc_test
    
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
   
    parser.add_argument("-u", "--trained_models",
                      help="Trained models list in CSV file.",
                      type=str, default="trained_models.csv")
    
    parser.add_argument("-b", "--batch_size",
                      help="The size of each batch for training.",
                      type=int, default=50)

    parser.add_argument("-p", "--epochs",
                      help="The number of epochs for training.",
                      type=int, default=5)
    
    parser.add_argument("-l", "--validation_samples",
                      help="The number of samples for validation.",
                      type=int, default=500)
    
    parser.add_argument("-c", "--dropout_rate",
                      help="Dropout Rate.",
                      type=int, default=0.20)
    
    parser.add_argument("-m", "--model",
                      help="Model to train.",
                      type=str, 
                      choices = ["cnn", "bgru", "c-bgru", "blstm", "c-blstm",
                                 "a-cnn", "a-blstm", "a-bgru", 
                                 "c-a-blstm", "c-a-bgru", "mac-blstm", 
                                 "mac-bgru", "all"],
                      default="all")
    
    parser.add_argument("-n", "--mode", help="Train or test model",
                      type=str, choices = ["train", "test"], default="train")

    args = parser.parse_args()
    model_list = None
    if args.model == "all":
        model_list = ["cnn", "bgru", "c-bgru", "a-bgru", 
                      "c-a-bgru", "blstm", "c-blstm", "a-blstm", 
                      "c-a-blstm"]
    else:
        model_list = [args.model]
    
    if args.mode == "train":    
        qc_train.train_model(model_list, args.trained_models,
                args.train_dataset, args.test_dataset, 
                args.embedding_file, args.attention_words_dictionary,
                args.batch_size, args.epochs, args.validation_samples,
                args.dropout_rate)
    else:
       qc_test.test_model(args.trained_models, args.train_dataset,
                args.test_dataset, args.embedding_file,
                args.attention_words_dictionary,
                args.batch_size)
    