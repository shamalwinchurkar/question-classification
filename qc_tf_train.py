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
import qc_tf_a_cnn


from sklearn.metrics import confusion_matrix

evaluate_every = 100
checkpoint_every = 100
            
def train_model(model_list, trained_models_file, train_dataset, val_dataset,
                test_dataset, embedding_file, atten_words_dict_file,
                batch_size, epochs, validation_samples, dropout_rate):
    
    ds = qc_data.Dataset(train_dataset, val_dataset, test_dataset, \
                         atten_words_dict_file)
        
    x_train, y_train, x_val, y_val, x_test, y_test, x_train_atten, \
        x_val_atten, x_test_atten = ds.load_with_val_dataset()
    
    emb = qc_emb.Embeddings(embedding_file)
    emb_matrix = emb.get_emb_matrix(ds.vocabulary_inv)
    emb_dim = emb.get_emb_dim()
    voc_size = ds.get_voc_size()
    num_class = ds.get_num_class()
    sen_len = x_train.shape[1]
    atten_sen_len = x_train_atten.shape[1]
    
    print("atten_sen_len = ", atten_sen_len)  
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
    
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = qc_tf_a_cnn.ACNN(emb_dim, voc_size, sen_len, atten_sen_len,
                                   num_class, emb_matrix, dropout_rate, atten_enable=True)
            
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 1.0, staircase=True)
            #optimizer = tf.train.AdamOptimizer(1e-3)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            run_folder = 'cnn_run'
            out_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())
        
            # Restore variable
            #saver.restore(sess, '')
        
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, x_atten_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_atten_x: x_atten_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dropout_rate
                        }
                _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                cnn.W = tf.clip_by_norm(cnn.W, 3)
                print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, x_atten_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_atten_x: x_atten_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                        }
                step, summaries, loss, accuracy, pred = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.pred],
                        feed_dict)
                print("VALID step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                y_test_nc = [ np.argmax(t) for t in y_batch ]
                cfm = confusion_matrix(y_test_nc, pred)
                print("Confusion Matrix:")
                print(cfm)
                cm_accuracy = 0
                length = len(cfm[0])
                for i in range(length):
                    cm_accuracy += cfm[i][i]
                cm_accuracy = (cm_accuracy / len(y_test_nc)) * 100
                cm_loss = 100 - cm_accuracy
                print("\nCM Accuracy {}".format(cm_accuracy))
                print("\nCM Loss {}".format(cm_loss))
            
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy, loss
                       
            # Generate batches
            batches = ds.batch_iter(list(zip(x_train, x_train_atten, y_train)),
                                         batch_size, epochs)

            # Training loop. For each batch...
            max_acc = 0
            best_at_step = 0
            for batch in batches:
                x_batch, x_atten_batch, y_batch = zip(*batch)
                train_step(x_batch, x_atten_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    acc, loss = dev_step(x_val, x_val_atten, y_val, writer=dev_summary_writer)
                    if acc >= max_acc:
                        if acc >= max_acc: max_acc = acc
                        best_at_step = current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                if current_step % checkpoint_every == 0:
                    print("Best of valid = {}, at step {}".format(max_acc, best_at_step))

            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            acc, loss = dev_step(x_test, x_test_atten, y_test, writer = None)
            print("Finish training. On test set: ", acc, loss) 
        