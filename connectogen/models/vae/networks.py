"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we
encourage you to explore architectures that fit your needs.
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, BatchNormalization, Conv3DTranspose
from keras.layers import LeakyReLU, Reshape, Lambda, Add, Flatten, Dense, Reshape, Permute
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
# from tf_elasticdeform import deform_grid
# import elasticdeform.tf as etf
# import tensorflow_probability as tfp #need to install 0.4.0 if tf version is 1.10.0
from group_norm import GroupNormalization


# other vm functions
import losses
import numpy as np



def fc_vae(vol_size, enc_nf, dec_nf, latent_dim = 256):
    # ndims = 1
    # upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    # pooling_layer = getattr(KL,'MaxPooling%dD' % ndims)
    # deconv_layer = getattr(KL,'Conv%dDTranspose' % ndims)
    # Conv = getattr(KL, 'Conv%dD' % ndims)
    # inputs
    input = Input(shape=vol_size)
    x = input
    print(enc_nf)
    print(dec_nf)
    for layer_dim in enc_nf:
        x = Dense(layer_dim,activation='relu')(x)

    mu = Dense(latent_dim,name = 'mu')(x)
    log_sigma = Dense(latent_dim,name = 'log_sigma')(x)
    z_params = concatenate([mu,log_sigma])
    x = Sample(name="z_sample")([mu, log_sigma])

    for layer_dim in dec_nf[:-1]:
        x = Dense(layer_dim,activation='relu')(x)

    x = Dense(dec_nf[-1])(x)

    outputs = [x, z_params]
    # build the model
    model = Model(inputs=[input], outputs=outputs)


    return model

# Helper functions
def conv_block(x_in, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """


    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    elif with_gn:
        def lambda_bn(x, groups):

            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = GroupNormalization(groups=groups, axis=channel_axis)(x)
            return x
    else:
        def lambda_bn(x):
            return x


    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = x_in
    x_out = Conv(nf, kernel_size=kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_out)
    if with_gn:
        x_out = lambda_bn(x_out, nf//2)
    else:
        x_out = lambda_bn(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def trans_block(x_in, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 1):
    """
    specific convolution module including convolution followed by leakyrelu
    """


    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    elif with_gn:
        def lambda_bn(x, groups):

            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = GroupNormalization(groups=groups, axis=channel_axis)(x)
            return x
    else:
        def lambda_bn(x):
            return x


    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    sdrop = getattr(KL, 'SpatialDropout%dD' % ndims)
    x_out = x_in
    x_out = Conv(nf, kernel_size=kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_out)
    if with_gn:
        x_out = lambda_bn(x_out, nf//2)
    else:
        x_out = lambda_bn(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    # x_out = sdrop(0)(x_out)
    return x_out

def dense_block(inputs, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 3):

    concatenated_inputs = inputs
    #nvidia-like is 3layers 4nb filters
    for i in range(3):
        x = conv_block(concatenated_inputs, 4, strides = strides, with_bn = with_bn, with_gn = with_gn, kernel_size = kernel_size)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=-1)
    # concatenated_inputs = conv_block(concatenated_inputs, nf, strides = strides, with_bn = with_bn, with_gn = with_gn, kernel_size = kernel_size)
    concatenated_inputs = trans_block(concatenated_inputs, nf, with_bn = with_bn, with_gn = with_gn)
    return concatenated_inputs

def res_shake(inputs, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 3):

    x1 = res_block(inputs,nf,strides=strides, with_bn = with_bn, with_gn = with_gn, kernel_size = kernel_size)
    x2 = res_block(inputs,nf,strides=strides, with_bn = with_bn, with_gn = with_gn, kernel_size = kernel_size)
    return Add()([inputs, ShakeShake()([x1,x2])])


# Helper functions
def nrc_conv_block(x_in, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """


    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    elif with_gn:
        def lambda_bn(x, groups):

            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = GroupNormalization(groups=groups, axis=channel_axis)(x)
            return x
    else:
        def lambda_bn(x):
            return x


    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = x_in
    if with_gn:
        x_out = lambda_bn(x_out, nf//2)
    else:
        x_out = lambda_bn(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    x_out = Conv(nf, kernel_size=kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_out)
    return x_out

def res_block(inputs, nf, strides=1,with_bn = False, with_gn = False, kernel_size = 3):

    x = inputs
    for i in range(2):
        x = nrc_conv_block(x, nf, strides = strides, with_bn = with_bn, with_gn = with_gn, kernel_size = kernel_size)

    x = Add()([x,inputs])

    return x

def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma) * noise
    return z


class Sample(Layer):
    """
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Negate(Layer):
    """
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape
