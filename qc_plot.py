# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:14:04 2019

@author: swinchurkar
"""
import os
import numpy as np
from matplotlib import pyplot
import qc_utils

color_list = ['red', 'green', 'blue', 'orange', 'yellow', 'black', 'brown',
              'cyan', 'magenta']


def plot_graphs(logs_dir, model_list, model_history, model_acc, model_loss,
                epochs, epoch_list, model_val_acc, model_val_loss):
    plot_model_list = []
    color = dict()
    i = 0    
    
    for model_name, history in model_history.items(): 
        plot_model_list.append(model_name.upper())
        color[model_name] = color_list[i]
        i += 1
        
        acc_fig = pyplot.figure()
        title = 'Accuracy Curves : ' + model_name.upper()
        pyplot.title(title, fontsize=18)
        pyplot.plot(history.history['acc'], 'b', linewidth=3.0)
        pyplot.plot(history.history['val_acc'], 'r', linewidth=3.0)
        pyplot.legend(['Training', 'Validation'], fontsize=12)
        pyplot.xlabel('Epochs ',fontsize=18)
        pyplot.ylabel('Accuracy',fontsize=18)
        pyplot.xlim(0, epochs)
        pyplot.xticks(np.arange(1, epochs + 1, 1.0))
        filename = os.path.join(logs_dir, model_name +'-accuracy.png')
        acc_fig.savefig(filename)
        pyplot.show()

        loss_fig = pyplot.figure()
        title = 'Loss Curves : ' + model_name.upper()
        pyplot.title(title, fontsize=18)
        pyplot.plot(history.history['loss'], 'b',linewidth=3.0)
        pyplot.plot(history.history['val_loss'], 'r',linewidth=3.0)
        pyplot.legend(['Training', 'Validation'], fontsize=12)
        pyplot.xlabel('Epochs ',fontsize=18)
        pyplot.ylabel('Loss',fontsize=18)
        pyplot.xlim(0, epochs)
        pyplot.xticks(np.arange(1, epochs + 1, 1.0))
        filename = os.path.join(logs_dir, model_name +'-loss.png')
        loss_fig.savefig(filename)
        pyplot.show()
        
    # Models Preformace Stastatic
            
    # Plot Trainning Parameters
    fig = pyplot.figure()
    pyplot.title('Training Accuracy Curves', fontsize=18)
    for model_name, history in model_history.items():
        pyplot.plot(history.history['acc'], c = color[model_name], linewidth=3.0)
        
    pyplot.legend(plot_model_list, fontsize=12)    
    pyplot.xlabel('Epochs ', fontsize=18)
    pyplot.ylabel('Accuracy', fontsize=18)
    pyplot.xlim(0, epochs)
    pyplot.xticks(np.arange(1, epochs + 1, 1.0))
    filename = os.path.join(logs_dir, 'training-accuracy.png')
    fig.savefig(filename)
    pyplot.show()

    fig = pyplot.figure()
    pyplot.title('Training Loss Curves', fontsize=18)
    for model_name, history in model_history.items():
        pyplot.plot(history.history['loss'], c = color[model_name], linewidth=3.0)
            
    pyplot.legend(plot_model_list, fontsize=12)    
    pyplot.xlabel('Epochs ',fontsize=18)
    pyplot.ylabel('Loss',fontsize=18)
    pyplot.xlim(0, epochs)
    pyplot.xticks(np.arange(1, epochs + 1, 1.0))
    filename = os.path.join(logs_dir, 'training-loss.png')
    fig.savefig(filename)
    pyplot.show()
    
    # Plot Validation Parameters
    fig = pyplot.figure()
    pyplot.title('Validation Accuracy Curves', fontsize=18)
    for model_name, history in model_history.items():
        pyplot.plot(history.history['val_acc'], c = color[model_name], linewidth=3.0)
        
    pyplot.legend(plot_model_list, fontsize=12)    
    pyplot.xlabel('Epochs ',fontsize=18)
    pyplot.ylabel('Accuracy',fontsize=18)
    pyplot.xlim(0, epochs)
    pyplot.xticks(np.arange(1, epochs + 1, 1.0))
    filename = os.path.join(logs_dir, 'validation-accuracy.png')
    fig.savefig(filename)
    pyplot.show()

    fig = pyplot.figure()
    pyplot.title('Validation Loss Curves', fontsize=18)
    for model_name, history in model_history.items():
        pyplot.plot(history.history['val_loss'], c = color[model_name], linewidth=3.0)
        
    pyplot.legend(plot_model_list, fontsize=12)
    pyplot.xlabel('Epochs ',fontsize=18)
    pyplot.ylabel('Loss',fontsize=18)
    pyplot.xlim(0, epochs)
    pyplot.xticks(np.arange(1, epochs + 1, 1.0))
    filename = os.path.join(logs_dir, 'validation-loss.png')
    fig.savefig(filename)
    pyplot.show()
        
    class_fig = pyplot.figure()
    pyplot.title('Test Accuracy Graph (Confustion Matrix)', fontsize=18)
    pyplot.ylim(0.0, 1.0)
    pyplot.bar(plot_model_list, model_acc,  
               width = 1.0, color = color_list) 
    pyplot.xlabel('Model', fontsize=18) 
    pyplot.ylabel('Accuracy (%)', fontsize=18) 
    
    filename = os.path.join(logs_dir, 'test-accuracy.png')
    class_fig.savefig(filename)
    pyplot.show()
    
    class_fig = pyplot.figure()
    pyplot.ylim(0.0, 1.0)
    pyplot.bar(plot_model_list, model_loss,  
               width = 1.0, color = color_list) 
    pyplot.xlabel('Model', fontsize=18) 
    pyplot.ylabel('Loss (%)', fontsize=18) 
    pyplot.title('Test Loss Graph (Confustion Matrix)', fontsize=18)
    filename = os.path.join(logs_dir, 'test-loss.png')
    class_fig.savefig(filename)
    pyplot.show()
    
    filename = os.path.join(logs_dir, 'test-results.csv')
    qc_utils.write_train_results(filename, plot_model_list, model_acc, model_loss,
                       epoch_list, model_val_acc, model_val_loss)
    
    return