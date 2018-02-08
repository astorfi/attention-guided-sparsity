import functools
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from losses import *
from sparsity_ops import *
FLAGS = None

def _group_lasso(v):
    # Group sparsity loss.
    group_loss_all = []
    for W in v:
        if 'bias' not in W.name:
            if 'conv' in W.name:
                # Input-channel-wise sparsity
                grouped_sum = tf.sqrt(tf.reduce_sum(tf.pow(W,2),axis=[0,1,2]))
                group_loss = tf.reduce_sum(grouped_sum)
                group_loss_all.append(group_loss)
            if 'fc' in W.name:
                # Input-channel-wise sparsity
                grouped_sum = tf.sqrt(tf.reduce_sum(tf.pow(W,2),axis=[0]))
                group_loss = tf.reduce_sum(grouped_sum)
                group_loss_all.append(group_loss)

    return tf.reduce_sum(group_loss_all)

def _group_variance(v):
    # Defining the group varianve function for attention-based sparsity.
    group_loss_variance = []
    for W in v:
        if 'bias' not in W.name:
            if 'conv' in W.name:
                grouped_elements = tf.reduce_sum(tf.pow(W,2),axis=[0,1,2])
                coefficient=1.0
                group_mean, group_variance = tf.nn.moments(grouped_elements, axes=[0])
                variance_loss = tf.divide(1.0, tf.divide(group_variance, coefficient))
                group_loss_variance.append(variance_loss)
            if 'fc' in W.name:
                grouped_elements = tf.reduce_sum(tf.pow(W,2),axis=[0])
                coefficient=1.0
                group_mean, group_variance = tf.nn.moments(grouped_elements, axes=[0])
                variance_loss = tf.divide(1.0, tf.divide(group_variance, coefficient))
                group_loss_variance.append(variance_loss)

    return tf.reduce_sum(group_loss_variance)
