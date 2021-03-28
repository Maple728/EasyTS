#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Maple.S
@project: EasyTS
@time: 2021/3/21 16:31
@desc:
"""
import tensorflow as tf

from .tf_utils import swap_axes


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

        # [n_heads, ..., n_queries, hidden_dim / n_heads]
        context_vector = scaled_dot_product_attention(queries, keys, values, key_mask, causality)
        # [..., n_queries, hidden_dim]
        context_vector = tf.concat(tf.unstack(context_vector, axis=0), axis=-1)

        # merge all outputs of each head
        output = layers.Dense(hidden_dim, name='head_merge')(context_vector)

        return output


def attention_layer(
        queries,
        keys,
        values,
        is_training,
        dropout_rate=0.0,
        score_method='dot',
        mask_method=None,
        name='att_layer'
):
    """

    Args:
        queries: Tensor with shape [..., n_q, dim]
        keys: Tensor with shape [..., n_k, dim]
        values: Tensor with shape [..., n_k, dim]
        is_training: bool
            Identify whether it is in training.
        dropout_rate: float, default 0.0
            Dropout rate of attention weights.
        score_method: str, default 'dot', option in ['add', 'dot', 'multiply']
            The method calculating score.
        mask_method: str, default None, option in [None, 'causality', 'causality-exclude']
            The method for masking attention weights.
        name: str
            Scope name.

    Returns:
        Context vector with shape [..., n_q, dim], derived from weighted values.
    """
    mask_val = -2 ** 15

    dim = queries.get_shape().as_list()[-1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if score_method == 'add':
            # shape -> [..., n_q, 1]
            q = tf.layers.dense(queries, 1) / (dim ** 0.5)
            # shape -> [..., n_k, 1]
            k = tf.layers.dense(keys, 1) / (dim ** 0.5)
            # shape -> [..., 1, n_k]
            k = swap_axes(k, -1, -2)

            # shape -> [..., n_q, n_k]
            scores = q + k
        elif score_method == 'dot':
            # shape -> [..., n_q, n_k]
            scores = tf.matmul(queries, keys, transpose_b=True) / (dim ** 0.5)
        elif score_method == 'multiply':
            scores = tf.layers.dense(queries, dim)
            # shape -> [..., n_q, n_k]
            scores = tf.matmul(scores, keys, transpose_b=True)
        else:
            raise RuntimeError(f'Unknown score_method: {score_method}')

        if mask_method is None:
            # shape -> [..., n_q, n_k]
            att_weights = tf.nn.softmax(scores, axis=-1)
        else:
            # use mask in attention weights
            # shape -> [..., n_q, n_k]
            ones_mat = tf.ones_like(scores)
            zeros_mat = tf.zeros_like(scores)
            mask_val_mat = ones_mat * mask_val

            tril_mat = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            if mask_method == 'causality':
                # keep the past and itself data, and drop the future data.
                # mask for weights calculation
                scores = tf.where(
                    tf.equal(tril_mat, 0),
                    mask_val_mat,
                    scores
                )
                att_weights = tf.nn.softmax(scores, axis=-1)

                # strict mask
                # att_weights = tf.where(
                #     tf.equal(tril_mat, 0),
                #     zeros_mat,
                #     att_weights
                # )
            elif mask_method == 'causality-exclude':
                scores = tf.where(
                    tf.equal(tril_mat, 1),
                    mask_val_mat,
                    scores
                )
                att_weights = tf.nn.softmax(scores, axis=-1)
            else:
                raise RuntimeError(f'Unknown mask_method: {mask_method}')

        # dropout on weights
        att_weights = tf.layers.dropout(att_weights, rate=dropout_rate, training=is_training)

        # shape -> [..., n_q, dim]
        context_vector = tf.matmul(att_weights, values)

        return context_vector
