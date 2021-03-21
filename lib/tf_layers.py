#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Maple.S
@project: EasyTS
@time: 2021/3/21 16:31
@desc:
"""
import tensorflow as tf


def ffn_layer(
        x,
        dims,
        is_training,
        activation=tf.nn.relu,
        dropout_rate=0.0,
        use_last_activation=True,
        has_residual=False,
        name='ffn_layer'
):
    """
    Feed Forward Network with short-cut and batch normalization.
    Args:
        x: Tensor
        dims: list
            Dimension for each inner layer.
        is_training: bool
        activation: Function
            Activation function for each inner layer.
        dropout_rate: float
            Dropout rate for each inner layer.
        use_last_activation: bool, default True
            Identify whether use activation in output layer.
        has_residual: bool, default False.
            Identify whether to use shortcut and batch_norm in the output layer, which means that hidden dim of output
            layer must be the same as the dim of the input.
        name: str
            Scope name.

    Returns:

    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_origin = x
        for dim in dims[:-1]:
            x = tf.layers.dense(x, dim, activation=activation)
            x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)

        if has_residual:
            x = tf.layers.batch_normalization(x + x_origin, training=is_training)

        if use_last_activation:
            x = activation(x)

        return x


def se_layer(x, name='se_layer'):
    """
    Squeeze-and-Excitation block for feature selection.

    Args:
        x: Tensor with shape [..., n, dim]
        name: str
            Scope name.

    Returns:
        Tensor with same shape as x, and different weights are assigned to its values.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        n = x.get_shape().as_list()[-2]
        x_flat = tf.layers.flatten(x)
        gate = tf.layers.dense(x_flat, n, activation=tf.nn.sigmoid)

        return x * gate[..., None]


def multi_head_attention(
        queries,
        keys,
        values,
        n_heads,
        key_mask,
        causality,
        scope
):
    """ Split the input into n_heads heads, then calculate the context vector for each head, and merge all
    context vectors into output.
    :param queries: the query sequences. [..., n_queries, hidden_dim]
    :param keys: the key sequences. [..., n_keys, hidden_dim]
    :param values: the value sequences whose length is same as keys. [..., n_keys, hidden_dim]
    :param n_heads: the number of heads
    :param key_mask: mask for keys. [..., n_keys]
    :param causality: mask for queries. True or False
    :param scope: the variable scope name
    :return: context vector. [..., n_queries, hidden_dim]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        hidden_dim = queries.get_shape().as_list()[-1]
        # transform input
        queries = layers.Dense(hidden_dim, name='Q_dense')(queries)
        keys = layers.Dense(hidden_dim, name='K_dense')(keys)
        values = layers.Dense(hidden_dim, name='V_dense')(values)

        # split the whole input into the part input for each head
        # [n_heads, ..., n_queries, hidden_dim / n_heads]
        queries = tf.stack(tf.split(queries, n_heads, axis=-1), axis=0)
        # [n_heads, ..., n_keys, hidden_dim / n_heads]
        keys = tf.stack(tf.split(keys, n_heads, axis=-1), axis=0)
        # [n_heads, ..., n_keys, hidden_dim / n_heads]
        values = tf.stack(tf.split(values, n_heads, axis=-1), axis=0)

        # [n_heads, ..., n_queries, hidden_dim / n_heads]p
        context_vector = scaled_dot_product_attention(queries, keys, values, key_mask, causality)
        # [..., n_queries, hidden_dim]
        context_vector = tf.concat(tf.unstack(context_vector, axis=0), axis=-1)

        # merge all outputs of each head
        output = layers.Dense(hidden_dim, name='head_merge')(context_vector)

        return output