from __future__ import print_function

import tensorflow as tf

epsilon = 0.00001

def micro_precision(target, pred):
    tp = tf.reduce_sum(target*pred)
    fp = tf.reduce_sum((1 - target)*pred)
    #fn = tf.reduce_sum(target*(1 - pred))
    return (tp)/(tp+fp+epsilon)


def macro_precision(target, pred):
    tp = tf.reduce_sum(target*pred, reduction_indices=[0])
    fp = tf.reduce_sum((1 - target)*pred, reduction_indices=[0])
    #fn = tf.reduce_sum(target*(1 - pred), reduction_indices=[0])
    return tf.reduce_sum((tp)/(tp+fp+epsilon))/tf.reduce_sum(tf.to_float((tp + fp) > 0))


def micro_f1(target, pred):
    tp = tf.reduce_sum(target*pred)
    fp = tf.reduce_sum((1 - target)*pred)
    fn = tf.reduce_sum(target*(1 - pred))
    precision = (tp)/(tp+fp+epsilon)
    recall = (tp)/(tp+fn+epsilon)
    return 2/(1/precision + 1/recall)


def macro_f1(target, pred):
    tp = tf.reduce_sum(target*pred, reduction_indices=[0])
    fp = tf.reduce_sum((1 - target)*pred, reduction_indices=[0])
    fn = tf.reduce_sum(target*(1 - pred), reduction_indices=[0])
    precision = tf.reduce_sum((tp)/(tp+fp+epsilon))/tf.reduce_sum(tf.to_float((tp + fp) > 0))
    recall = tf.reduce_sum((tp)/(tp+fn+epsilon))/tf.reduce_sum(tf.to_float((tp + fn) > 0))
    return 2/(1/precision + 1/recall)
