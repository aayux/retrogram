import os
import gc
import fire

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from models.skipgram import SGNS
from utils.data_utils import DataLoader, batchify, iterate

n_epochs = 3
batch_size = 64
init_learning_rate = 1.
context_window = 1
n_sample = 256
mu = 1.2

def train(experiment, directory, path='./data'):
    loader = DataLoader(path, directory, experiment)
    loader.load_word2vec(filename='embeddings.npy')
    word_idxs, new_embeddings, slice_ = loader.prepare(top_k=500000)

    base_embeddings = new_embeddings[slice_:]

    vocab_size, embedding_dims = new_embeddings.shape
    old_vocab_size = vocab_size - slice_

    print(f'Size of vocabulary: {vocab_size}')

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, 
                                      log_device_placement=False)
        
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_conf.gpu_options.allow_growth = True
        
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            sgns = SGNS(vs=vocab_size, vs_=old_vocab_size, 
                        ems=embedding_dims, smpl=n_sample, mu=mu)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Define the optimizer
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 10000, 0.98, 
                                            staircase=True)
            
            # learning_rate = init_learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(sgns.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            runs_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', experiment))
            print(f'Writing to {runs_dir}\n')

            # Summaries for loss and retrofit penalty
            loss_summary = tf.summary.scalar('loss', sgns.loss)
            retro_summary = tf.summary.scalar('retrofit', sgns.penalty)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, retro_summary])
            train_summary_dir = os.path.join(runs_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            feed_dict = {
                sgns.old_em_placeholder: base_embeddings,
                sgns.em_placeholder: new_embeddings
            }
            
            sess.run([sgns.old_em_init, sgns.em_init], feed_dict)
            
            # kick in some garbage collection
            del base_embeddings, new_embeddings
            gc.collect()

            def batch_trainer(word, targ):
                r"""
                A single training step
                """
                feed_dict = {
                    sgns.word: word,
                    sgns.targ: targ
                }
                # _, step, summaries, _grad = sess.run([train_op, global_step, train_summary_op, 
                #                                       sgns.force_grad], feed_dict)
                _, step, summaries = sess.run([train_op, global_step, train_summary_op], feed_dict)
                train_summary_writer.add_summary(summaries, step)

            # print(tf.get_default_graph().as_graph_def())

            for epoch in tqdm(range(n_epochs)):
                
                # Generate batches
                batches = batchify(iterate(word_idxs, context_window), batch_size)
                
                # Training loop. For each batch.
                for batch in batches:
                    word, targ = batch
                    batch_trainer(word, targ)
                    current_step = tf.train.global_step(sess, global_step)
                
            # pull the layer weights and save embeddings as npy file
            embeddings = sess.run(sgns.we)
            
            if not os.path.exists(os.path.join(path, directory, 'embeddings')):
                os.makedirs(os.path.join(path, directory, 'embeddings'))
            
            np.save(os.path.join(path, directory, 'embeddings', f'{experiment}.npy'), embeddings)

if __name__ == '__main__':
    fire.Fire(train)
