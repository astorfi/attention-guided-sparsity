import functools
from apiclient import errors
from apiclient.http import MediaFileUpload
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from losses import *
from sparsity_ops import *
FLAGS = None


def train():

    # Import MNIST data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()
    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    with tf.name_scope('training_status'):
        training_status = tf.placeholder(tf.bool)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

    def conv2d(x, W, padding='SAME'):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def nn_layer(input_tensor, input_dim, output_dim, layer_name, training_status=True,\
                 act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])

                # Get the general summaries
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):

                # At evaluation time, some weights are forced to be zero with the sparsity criterion.
                weights = tf.cond(training_status,
                   true_fn = lambda: _sparse_fn(weights,threshold=0.0),
                   false_fn = lambda: _sparse_fn(weights,threshold=FLAGS.sparsity_threshold))

                preactivate = tf.matmul(input_tensor, weights) + biases
                # tf.summary.histogram('pre_activations', preactivate)

            # Activation summary
            activations = act(preactivate, name='activation')
            tf.summary.scalar('sparsity', tf.nn.zero_fraction(activations))
            tf.summary.histogram('activations', activations)


            # Overall neurons
            neurons = tf.reduce_sum(tf.abs(weights), axis=1)
            tf.summary.histogram('neurons', neurons)


            return activations

    def nn_conv_layer(input_tensor, w_shape, b_shape, layer_name, padding='SAME', \
                      training_status=True, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable(w_shape)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable(b_shape)
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):

                # At evaluation time, some weights are forced to be zero with the sparsity criterion.
                weights = tf.cond(training_status,
                   true_fn = lambda: _sparse_fn(weights,threshold=0.0),
                   false_fn = lambda: _sparse_fn(weights,threshold=FLAGS.sparsity_threshold))

                preactivate = conv2d(input_tensor, weights,padding) + biases
                # tf.summary.histogram('pre_activations', preactivate)

            activations = act(preactivate, name='activation')
            # tf.summary.histogram('activations', activations)
            return activations

    def net(x,training_status):

        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = nn_conv_layer(x_image, [5, 5, 1, 64], [64], 'conv1', \
                                training_status=training_status, act=tf.nn.relu)

        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = nn_conv_layer(h_pool1, [5, 5, 64, 128], [128], 'conv2',\
                                training_status=training_status, act=tf.nn.relu)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])

        h_fc1 = nn_layer(h_pool2_flat, 7 * 7 * 128, 512, 'fc1', \
                         training_status=training_status, act=tf.nn.relu)
        dropped_h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

        h_fc2 = nn_layer(dropped_h_fc1, 512, 256, 'fc2', \
                         training_status=training_status, act=tf.nn.relu)
        dropped_h_fc2 = tf.nn.dropout(h_fc2, keep_prob)

        # Do not apply softmax activation yet, see below.
        output = nn_layer(dropped_h_fc2, 256, 10, 'softmax', \
                          training_status=training_status, act=tf.identity)

        return output, keep_prob

    # Network
    output, keep_prob = net(x,training_status)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=y_, logits=output)
    tf.summary.scalar('cross_entropy', cross_entropy)

    #############################
    ########## LOSS #############
    #############################

    # Get all trainable variables except biases
    trainable_variables = tf.trainable_variables()

    # Compute the regularization term
    with tf.name_scope('group_lasso'):
        lasso_loss = 0.001 * _group_lasso(trainable_variables)

    with tf.name_scope('group_variance'):
        variance_loss = 0.01 * _group_variance(trainable_variables)

    tf.losses.add_loss(
        lasso_loss,
        loss_collection=tf.GraphKeys.LOSSES
    )

    tf.losses.add_loss(
        variance_loss,
        loss_collection=tf.GraphKeys.LOSSES
    )

    # Compute the regularization term
    with tf.name_scope('Sparsity'):
        sparsity, num_params = _sparsity_calculatior(trainable_variables)

    tf.summary.scalar('sparsity', sparsity)

    ###############################################
    ############### Total Loss  ###################
    ###############################################

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')
    list_losses = tf.losses.get_losses(loss_collection=tf.GraphKeys.LOSSES)
    reg_losses = tf.losses.get_losses(loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)


    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            total_loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(output, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            is_train = True
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            is_train = False
            xs, ys = mnist.test.next_batch(1000)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k, training_status:is_train}

    for i in range(1, FLAGS.max_steps):

        if i % 100 == 0:  # Record summaries and test-set accuracy
            summary, acc,sparsity_value, num_parameters = sess.run([merged, \
                       accuracy,sparsity, num_params], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy and Sparsity at step %s: %s , %s\n number of parameters= %s' % (i, \
                                                           acc, sparsity_value,num_parameters))


        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
