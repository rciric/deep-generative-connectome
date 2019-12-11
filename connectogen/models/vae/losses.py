"""
losses for VoxelMorph
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
from tensorflow.python.ops import math_ops

def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.arange(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


class Losses():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, flow_vol_shape=None):

        self.flow_vol_shape = flow_vol_shape

    def recon_loss(self, y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))


    def kl_loss(self, _, y_pred):

        mean = y_pred[..., :y_pred.get_shape()[-1]//2]
        log_sigma = y_pred[..., y_pred.get_shape()[-1]//2:]
        # loss = 1 + log_sigma - K.square(mean) - K.exp(log_sigma)
        # loss = -1/2*K.sum(loss, axis=-1)
        loss = - 0.5 * K.mean(1 + 2*log_sigma - K.square(mean) - K.square(K.exp(log_sigma)), axis=-1)
        return loss
