# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:58:11 2019

@author: swinchurkar
"""

def write_train_results(filename, plot_model_list, model_acc, model_loss,
                       epoch_list, model_val_acc, model_val_loss):
    texts = "MODEL," + ','.join(str(e) for e in plot_model_list)
    texts += "\nEPOCH," + ','.join(str(e) for e in epoch_list)
    texts += "\nVALIDATION ACCURACY," + ','.join(str(e) for e in model_val_acc)
    texts += "\nVALIDATION LOSS," + ','.join(str(e) for e in model_val_loss)
    texts += "\nTEST ACCURACY," + ','.join(str(e) for e in model_acc)
    texts += "\nTEST LOSS," + ','.join(str(e) for e in model_loss)             
    file = open(filename, "w", encoding = 'utf-8')
    file.write(texts)
    file.close()

def write_test_results(filename, model_list, model_acc, model_loss):
    texts = "MODEL," + ','.join(str(e) for e in model_list)
    texts += "\nTEST ACCURACY," + ','.join(str(e) for e in model_acc)
    texts += "\nTEST LOSS," + ','.join(str(e) for e in model_loss)             
    file = open(filename, "w", encoding = 'utf-8')
    file.write(texts)
    file.close()

def read_model_val_cfg(filename):
    epoch = None
    acc = None 
    loss = None
    file = open(filename, "r", encoding = 'utf-8')
    for line in file:
        words = line.split(',')
        if words[0] == "EPOCH":
            epoch = int(words[1])
        elif words[0] == "ACCURACY":
            acc = float(words[1])
        elif words[0] == "LOSS":
            loss = float(words[1])
    file.close()        
    return epoch, acc, loss

def read_model_cfg(trained_models_file):
    model_dict = dict()
    file = open(trained_models_file, "r", encoding = 'utf-8')
    for line in file:
        words = line.split(',')
        model_dict[words[0]] = words[1]
    file.close()        
    return model_dict

def write_model_cfg(trained_models_file, model_dict):
    text = str()
    file = open(trained_models_file, "w", encoding = 'utf-8')
    for model_name, model_file in model_dict.items():
        text += model_name + "," + model_file + "\n"
    file.write(text)    
    file.close()        
    return model_dict