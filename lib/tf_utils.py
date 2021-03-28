#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: AEHN
@time: 2019/12/22 21:49
@desc:
"""
from functools import reduce
from operator import mul
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


FLOAT_TYPE = tf.float32


# ---------------------------- General Functions -----------------------------
def get_num_trainable_params():
    """
    Get the number of trainable parameters in current session (model).

    Returns:
        Number of trainable parameters of model.
    """
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def tensordot(tensor_a, tensor_b):
    """
    Tensor dot function. The last dimension of tensor_a and the first dimension of tensor_b must be the same.

    Args:
        tensor_a: Tensor
            Shape is [..., dim_a]
        tensor_b: Tensor
            Shape is [dim_b, ...]

    Returns:
        Tensor with shape [..., ...].
    """
    last_idx_a = len(tensor_a.get_shape().as_list()) - 1
    return tf.tensordot(tensor_a, tensor_b, [[last_idx_a], [0]])


def label_smoothing(x_onehot, epsilon=0.1):
    """
    Label smoothing for one-hot type label.

    Args:
        x_onehot: Tensor with shape [..., depth]
        epsilon: float, default 0.1
            The smoothing factor in range [0.0, 1.0].

    Returns:
        Tensor with same shape as x_onehot, and its value has been smoothed.
    """
    depth = x_onehot.get_shape().as_list()[-1]
    return (1 - epsilon) * x_onehot + epsilon / depth


def swap_axes(tensor, axis1, axis2):
    """
    Interchange two axes of an tensor.
    eg.
        Ten rank of source tensor is 3, the axis1 and axis2 is 2 and 1 respectively, then the
        returned tensor's dimension is [0, 2, 1].

    Args:
        tensor: Tensor
            The source tensor.
        axis1: int
            First axis, less than rank(tensor).
        axis2: int
            Second axis, less than rank(tensor).

    Returns:
        A Tensor with same shape as tensor, and its axes has been swapped.
    """
    rank = len(tensor.shape.as_list())
    tensor_perm = list(range(rank))
    axis1 = (axis1 + rank) % rank
    axis2 = (axis2 + rank) % rank

    tensor_perm[axis1] = axis2
    tensor_perm[axis2] = axis1

    return tf.transpose(tensor, perm=tensor_perm)


def create_tensor(shape, fill_value):
    """
    Creates a tensor with all elements set to value and dtype is same as fill_value.

    Args:
        shape: list
            Shape of target tensor.
        fill_value: int, float or other type

    Returns:
        Tensor with shape "shape" whose value is fill_value.
    """
    return tf.fill(tf.stack(shape), fill_value)


def layer_norm(inputs, epsilon=1e-8, name='layer_norm'):
    """
    Layer Normalization.
    Args:
        inputs: Tensor
        epsilon: float, default 1e-8
        name: str
            Scope name.

    Returns:
        A tensor with same shape and data type as 'inputs'.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        param_shape = inputs.get_shape()[-1:]
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', param_shape, initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        inputs_norm = (inputs - mean) / (variance + epsilon ** 0.5)
        output = gamma * inputs_norm + beta

        return output
# ----------------------------------------------------------------------------


# --------------------------- Activation Functions ---------------------------
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def prelu(x, name='prelu'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alpha = tf.get_variable(
            'alpha',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0)
        )
        w1 = 0.5 * (1 + alpha)
        w2 = 0.5 * (1 - alpha)

        return w1 * x + w2 * tf.nn.relu(x)


def dice(x, is_training, name='dice'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alpha = tf.get_variable(
            'alpha',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0)
        )
        x_norm = tf.layers.batch_normalization(x, epsilon=1e-8, training=is_training)
        p = tf.nn.sigmoid(x_norm)

        return p * x + (1 - p) * x * alpha
# ----------------------------------------------------------------------------


def get_variable_weights(name, shape, collections=None):
    return tf.get_variable(name, shape=shape, dtype=FLOAT_TYPE,
                           initializer=tf.glorot_normal_initializer(),
                           collections=collections)


def get_variable_bias(name, shape, collections=None):
    return tf.get_variable(name, shape=shape, dtype=FLOAT_TYPE,
                           initializer=tf.constant_initializer(0.1),
                           collections=collections)





class Attention:
    def __init__(self, cell_units, score_method='general'):
        self.cell_units = cell_units

        if score_method == 'general':
            attention_w1 = layers.Dense(self.cell_units, name='w1')
            attention_w2 = layers.Dense(self.cell_units, name='w2')
            attention_v = layers.Dense(1, name='v')
            score_fn = lambda q, k: attention_v(tf.nn.tanh(attention_w1(q) + attention_w2(k))) / tf.sqrt(
                tf.cast(self.cell_units, dtype=tf.float32))
        elif score_method == 'exp_dot':
            score_fn = lambda q, k: tf.reduce_sum(tf.exp(q * k), axis=-1, keepdims=True)
        else:
            raise RuntimeError('Unknown score_method:' + score_method)
        self.score_fn = score_fn

    def compute_attention_weight(self, queries, keys, values, pos_mask=None):
        """
        :param queries: (batch_size, n_queries, hidden_dim)
        :param keys: (batch_size, n_keys, hidden_dim)
        :param values: (batch_size, n_values, hidden_dim)
        :param pos_mask: ['self-right', 'right', None]
        self-right: mask values for the upper right area, excluding the diagonal
        right: mask values for the upper right area, including the diagonal
        None: no mask.
        :return: (batch_size, num_queries, cell_units), (batch_size, num_queries, num_keys, 1)
        """
        MASKED_VAL = - 2 ** 32 + 1
        # (batch_size, num_queries, 1, hidden_dim)
        q = tf.expand_dims(queries, axis=2)
        # (batch_size, 1, num_keys, hidden_dim)
        k = tf.expand_dims(keys, axis=1)
        v = tf.expand_dims(values, axis=1)

        # (batch_size, num_queries, num_keys, 1)
        score = self.score_fn(q, k)

        if pos_mask:
            # (batch_size, num_queries, num_keys)
            score = tf.squeeze(score, axis=-1)

            ones_mat = tf.ones_like(score)
            zeros_mat = tf.zeros_like(score)
            masked_val_mat = ones_mat * MASKED_VAL

            # (batch_size, num_queries, num_keys)
            lower_diag_masks = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            if pos_mask == 'right':
                # mask values for the upper right area, including the diagonal
                # (batch_size, num_queries, num_keys)
                score = tf.where(tf.equal(lower_diag_masks, 0),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_diag_masks, 0),
                                            zeros_mat,
                                            attention_weight)
            elif pos_mask == 'self-right':
                # mask values for the upper right area, excluding the diagonal
                # transpose to upper triangle
                lower_masks = tf.transpose(lower_diag_masks, perm=[0, 2, 1])

                score = tf.where(tf.equal(lower_masks, 1),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_masks, 1),
                                            zeros_mat,
                                            attention_weight)

            else:
                raise RuntimeError('Unknown pas_mask: {}'.format(pos_mask))

            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.expand_dims(attention_weight, axis=-1)
        else:
            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.nn.softmax(score, axis=2)

        # (batch_size, num_queries, num_keys, cell_units)
        context_vector = attention_weight * v

        # (batch_size, num_queries, cell_units)
        context_vector = tf.reduce_sum(context_vector, axis=2)

        return context_vector, attention_weight
