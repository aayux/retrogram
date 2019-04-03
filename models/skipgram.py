import tensorflow as tf
import numpy as np

from tensorflow.python.framework import function

@function.Defun(tf.float32, tf.float32)
def gradient_of_norm(x, dy):
    return dy * (x / (tf.norm(x) + 1e-19))

@function.Defun(tf.float32, grad_func=gradient_of_norm)
def norm(x):
    return tf.norm(x)

class SGNS(object):
    r"""
    A skip-gram negative sampling (SGNS) model for generating word vectors.
    """
    def __init__(self, vs, vs_, ems, smpl, mu):

        # Placeholders for words and context
        self.word = tf.placeholder(tf.int32, shape=[None], name='word')
        self.targ = tf.placeholder(tf.int32, shape=[None, 1], name='targets')

        with tf.device('/cpu:0'):
            self.we = tf.Variable(tf.zeros([vs, ems]), name='embedding-matrix')
            self.old_we = tf.Variable(tf.zeros([vs_, ems]), trainable=False)
            
            self.old_em_placeholder = tf.placeholder(tf.float32, [vs_, ems])
            self.old_em_init = self.old_we.assign(self.old_em_placeholder)

            self.em_placeholder = tf.placeholder(tf.float32, [vs, ems])
            self.em_init = self.we.assign(self.em_placeholder)
            
            # Embedding layer
            with tf.name_scope('hidden-layer'):
                self.char_em = tf.nn.embedding_lookup(self.we, self.word)
            # self.em_chars_expanded = tf.expand_dims(self.char_em, -1)
        
        with tf.name_scope('output-layer'):
            W = tf.get_variable('weight', shape=[vs, ems], 
                                initializer=tf.truncated_normal_initializer(
                                stddev=1. / (ems ** .5)))
            b = tf.Variable(tf.zeros(shape=[vs]), name='bias')

        # Calculate the NCE loss
        with tf.name_scope('loss'):
            losses = tf.nn.nce_loss(weights=W, biases=b, labels=self.targ, inputs=self.char_em, 
                                  num_sampled=smpl, num_classes=vs)
            
            self.penalty = self.retrofit(slice_=-vs_, metric='cosine')

            self.loss = tf.reduce_mean(losses) + (mu * self.penalty)

    def retrofit(self, slice_=None, metric='euclidean'):
        if metric == 'euclidean':
            difference = self.we[slice_:] - self.old_we
            distance = norm(difference)
            self.force_grad = tf.gradients(distance, [difference])
        elif metric == 'cosine':
            distance = tf.losses.cosine_distance(tf.nn.l2_normalize(self.we[slice_:], 0), 
                                                 tf.nn.l2_normalize(self.old_we, 0), dim=0)
        return distance
