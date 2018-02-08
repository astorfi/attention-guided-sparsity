import tensorflow as tf
import functools

def _sparse_fn(weights,threshold=0.001):
    # Force weights less than a threshold to zero.
    # Set threshold and clip
    # This force the weights in range [-threshold,threshold] to be zero.
    W_signed = tf.sign(weights)
    W_sparse_temp = tf.clip_by_value(tf.subtract(tf.abs(weights),threshold),\
                                     clip_value_min=0,clip_value_max=10000.0)
    W_sparse_signed = tf.sign(W_sparse_temp)
    W_sparse = W_signed * W_sparse_signed * tf.add(W_sparse_temp,threshold)
    return W_sparse


def _sparsity_calculatior(v,threshold=0.0):

    # Calculation of the sparsity of the network
    sparsity_layers = []
    num_params_layers = []
    for W in v:
          if 'bias' not in W.name:
              if 'conv' or 'fc' in W.name:

                  # Set threshold and clip
                  W_sparse = tf.clip_by_value(tf.subtract(tf.abs(W), threshold),\
                                              clip_value_min=0, clip_value_max=10000)

                  # Sparsity calculation
                  num_nonzero = tf.cast(tf.count_nonzero(W_sparse),tf.float32)
                  num_weights = functools.reduce(lambda x, y: x*y, W_sparse.get_shape())
                  non_sparsity_level = tf.divide(num_nonzero,tf.cast(num_weights,tf.float32))
                  Sparsity = tf.subtract(1.0,non_sparsity_level)

                  # Add the sparsity of each layer to the list
                  sparsity_layers.append(Sparsity)

                  # Add the number of parameters for each layer
                  num_params_layers.append(num_nonzero)


    return tf.reduce_mean(sparsity_layers), tf.reduce_sum(num_params_layers)
