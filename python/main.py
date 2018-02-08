from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from losses import *
from sparsity_ops import *
from train import *
FLAGS = None
# from IPython.core.debugger import set_trace


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train(FLAGS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=300000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--sparsity_threshold', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str,
       default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/input_data'),
       help='Directory for storing input data')
    parser.add_argument('--log_dir',type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
        'tensorflow/mnist/logs/mnist_sparsity'),help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
