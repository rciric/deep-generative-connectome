#   PLEASE INSTALL KERAS 2.2.4 OR GREATER
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Dense, concatenate, Dropout, SpatialDropout2D, MaxPooling3D, Conv3D, Conv3DTranspose
from keras.layers.core import Activation, Layer
from keras.layers.merge import add as keras_add
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K
from subtle_metrics import mixedLoss
from subtle_filter import multi_lateral_filter
from keras.models import model_from_yaml, model_from_json
from keras.regularizers import l2
from keras.utils import multi_gpu_model

import numpy as np

# clean up
def clearKerasMemory():
    K.clear_session()

# use part of memory
def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))

# load models
def loadKerasModel(filepath, string_model=None, format_export='json'):
    if string_model is None:
        with open(filepath,'r') as file:
            string_model = file.read()
    if format_export == 'json':
        model = model_from_json(string_model)
    else:
        model = model_from_yaml(string_model)        
    return model

# # u-net
# def unet(self):
#         """U-Net Generator"""

#         def conv2d(layer_input, filters, f_size=4, bn=True):
#             """Layers used during downsampling"""
#             d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
#             d = LeakyReLU(alpha=0.2)(d)
#             if bn:
#                 d = BatchNormalization(momentum=0.8)(d)
#             return d

#         def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
#             """Layers used during upsampling"""
#             u = UpSampling2D(size=2)(layer_input)
#             u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
#             if dropout_rate:
#                 u = Dropout(dropout_rate)(u)
#             u = BatchNormalization(momentum=0.8)(u)
#             u = Concatenate()([u, skip_input])
#             return u

#         # Image input
#         d0 = Input(shape=self.img_shape)

#         # Downsampling
#         d1 = conv2d(d0, self.gf, bn=False)
#         d2 = conv2d(d1, self.gf*2)
#         d3 = conv2d(d2, self.gf*4)
#         d4 = conv2d(d3, self.gf*8)
#         # d5 = conv2d(d4, self.gf*8)
#         # d6 = conv2d(d5, self.gf*8)
#         # d7 = conv2d(d6, self.gf*8)

#         # Upsampling
#         u1 = deconv2d(d7, d6, self.gf*8)
#         u2 = deconv2d(u1, d5, self.gf*8)
#         u3 = deconv2d(u2, d4, self.gf*8)
#         u4 = deconv2d(u3, d3, self.gf*4)
#         u5 = deconv2d(u4, d2, self.gf*2)
#         u6 = deconv2d(u5, d1, self.gf)
#         u7 = UpSampling2D(size=2)(u6)

#         u4 = deconv2d(d4, d3, self.gf*8)
#         u3 = deconv2d(u4, d2, self.gf*8)
#         u2 = deconv2d(u3, d1, self.gf*8)
#         u1 = deconv2d(u3, d1, self.gf*8)
#         u0 = UpSampling2D(size=2)(u1)
#         output_img = Conv2D(self.channels, kernel_size=4, strides=1,
#                           padding='same', activation='tanh')(u0)

#         return Model(d0, output_img)

# encoder-deocder


def get_unet(num_channel_input=1, num_channel_output=1,
                img_rows=128, img_cols=128,
                y=np.array([-1, 1]),  # change to output_range in the future
                output_range=None,
                lr_init=None, loss_function=mixedLoss(),
                metrics_monitor=[mean_absolute_error, mean_squared_error],
                num_poolings=3, num_conv_per_pooling=3, num_channel_first=32,
                with_bn=True,  # don't use for F16 now
                with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                with_multi_lateral_filter=False,  # guilded filter bank
                # filter parameters
                img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                # new settings
                activation_conv = 'relu',  # options: 'elu', 'selu'
                activation_output = None, #options: 'tanh', 'sigmoid', 'linear', 'softplus'
                kernel_initializer = 'zeros', # options: 'he_normal'
                verbose=1):
    inputs = Input((img_rows, img_cols, num_channel_input))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    optimizer = Adam(lr=lr_init)
    model.compile(loss=loss_function, optimizer=optimizer,metrics=metrics_monitor)
    return model

def get_winner2017unet(num_channel_input=1, num_channel_output=1,
                img_rows=128, img_cols=128,
                y=np.array([-1, 1]),  # change to output_range in the future
                output_range=None,
                lr_init=None, loss_function=mixedLoss(),
                metrics_monitor=[mean_absolute_error, mean_squared_error],
                num_poolings=3, num_conv_per_pooling=3, num_channel_first=32,
                with_bn=True,  # don't use for F16 now
                with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                with_multi_lateral_filter=False,  # guilded filter bank
                # filter parameters
                img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                # new settings
                activation_conv = 'relu',  # options: 'elu', 'selu'
                activation_output = None, #options: 'tanh', 'sigmoid', 'linear', 'softplus'
                kernel_initializer = 'zeros', # options: 'he_normal'
                verbose=1):
    inputs = Input((img_rows, img_cols, num_channel_input))
    conv_1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv_1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)

    conv_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)
    conv_2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_2_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)

    conv_3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_2)
    conv_3_2 = Conv2D(128, (4, 4), activation='relu', padding='same')(conv_3_1)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3_2)

    pool_3_1 = Dropout(0.5)(pool_3)
    conv_4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3_1)
    conv_4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_4_1)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4_2)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)

    decv_1 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(conv_5)
    ups_1 = UpSampling2D(size=(2,2))(decv_1)

    decv_2_1 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(ups_1)
    decv_2_2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decv_2_1)
    ups_2 = UpSampling2D(size=(2, 2))(decv_2_2)

    decv_3_1 = Conv2DTranspose(128, (4, 4), activation='relu', padding='same')(ups_2)
    decv_3_2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decv_3_1)
    ups_3 = UpSampling2D(size=(2, 2))(decv_3_2)

    decv_4_1 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(ups_3)
    decv_4_2 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(decv_4_1)
    ups_4 = UpSampling2D(size=(2, 2))(decv_4_2)

    ups_4_1 = Dropout(0.5)(ups_4)   
    decv_5_1 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(ups_4_1)
    out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decv_5_1)

    model = Model(inputs=[inputs], outputs=out)
    optimizer = Adam(lr=lr_init)
    model.compile(loss=loss_function, optimizer=optimizer,metrics=metrics_monitor)
    return model


def dropoutResUNet(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128,
                   y=np.array([-1, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function=mixedLoss(),
                   metrics_monitor=[mean_absolute_error, mean_squared_error],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,
                   with_bn=True,  # don't use for F16 now
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   with_multi_lateral_filter=False,  # guilded filter bank
                   # filter parameters
                   img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                   # new settings
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            x = BatchNormalization()(x)
            return Activation(activation_conv)(x)
    else:
        def lambda_bn(x):
            return x

    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)
    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # step1
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        conv1 = Conv2D(num_channel_first, (3, 3),
                       padding="same",
                       activation=activation_conv,
                       kernel_initializer=kernel_initializer)(conv1)

        conv1 = lambda_bn(conv1)
        if (i + 1) % 2 == 0 and i != 1:
            conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            conv_encoder = Conv2D(
                num_channel, (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoder)

            conv_encoder = lambda_bn(conv_encoder)
            if (j + 1) % 2 == 0 and j != 1:
                conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
                         kernel_initializer=kernel_initializer,
                         bias_initializer='zeros')(pool_encoders[-1])
    conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
                         kernel_initializer=kernel_initializer,
                         bias_initializer='zeros')(conv_center)

    # conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):
        # concate from encoding layers
        #        upsample_decoder = concatenate(
        #            [UpSampling2D(size=(2, 2))(conv_decoders[-1]), conv_encoders[-i]])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i]])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            conv_decoder = Conv2D(
                list_num_features[-i], (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoder)

            conv_decoder = lambda_bn(conv_decoder)
            if (j + 1) % 2 == 0 and j != 1:
                conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model



def nestedUNet(num_channel_input=1, num_channel_output=1,
                 img_rows=128, img_cols=128,
                 y=np.array([-1, 1]),  # change to output_range in the future
                 output_range=None,
                 lr_init=None, loss_function=mixedLoss(),
                 metrics_monitor=[mean_absolute_error, mean_squared_error],
                 num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,
                 with_bn=True,  # don't use for F16 now
                 with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                 with_multi_lateral_filter=False,  # guilded filter bank
                 # filter parameters
                 img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                 # new settings
                 activation_conv='relu',  # options: 'elu', 'selu'
                 activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                 kernel_initializer='zeros',  # options: 'he_normal'
                 verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            x = BatchNormalization()(x)
            return Activation(activation_conv)(x)
    else:
        def lambda_bn(x):
            return x

    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)

    # step1
    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    Added Dense connections in the concatenation part of the decoding side of the Unet
        See the additional nested for loop in the decoding side of the Unet
    '''
    '''
    Below was modified by Charles
    '''
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)

        conv1 = Conv2D(num_channel_first, (3, 3),
                       padding="same",
                       activation=activation_conv,
                       kernel_initializer=kernel_initializer)(conv1)
        conv1 = lambda_bn(conv1)
        if (i + 1) % 2 == 0 and i != 1:
            conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            conv_encoder = Conv2D(
                num_channel, (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoder)
            conv_encoder = lambda_bn(conv_encoder)
            if (j + 1) % 2 == 0 and j != 1:
                conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        # pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    conv_encoders.append(pool_encoders[-1])
    conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
                         kernel_initializer=kernel_initializer,
                         bias_initializer='zeros')(pool_encoders[-1])
    conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
                         kernel_initializer=kernel_initializer,
                         bias_initializer='zeros')(conv_center)

    #    conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    upsample_decoders = []
    middle_decoders = []
    for i in range(1, num_poolings + 1):
        # concate from encoding layers
        # upsample_decoder = UpSampling2D(size=(2, 2))(conv_decoders[-1])
        upsample_decoder = Conv2DTranspose(
            list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
            activation=activation_conv,
            kernel_initializer=kernel_initializer)(conv_decoders[-1])
        upsample_decoders.append(upsample_decoder)
        if i >= 2:
            middle_decoder = Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoders[-i])
            middle_decoders.append(middle_decoder)
            upsample_decoder = concatenate([upsample_decoder, middle_decoders[-1]])
            if i >= 3:
                for iter in range(1,i-1):
                    new_middle_decoder = Conv2DTranspose(
                    list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                    activation=activation_conv,
                    kernel_initializer=kernel_initializer)(middle_decoders[-i+1])
                    middle_decoders.append(new_middle_decoder)
                    upsample_decoder = concatenate([upsample_decoder, middle_decoders[-1]])

        upsample_decoder = concatenate(
                [upsample_decoder, conv_encoders[-i-1]])
        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            conv_decoder = Conv2D(
                list_num_features[-i], (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoder)

            conv_decoder = lambda_bn(conv_decoder)
            if (j + 1) % 2 == 0 and j != 1:
                conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        # conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model

def squeezeUNet(num_channel_input=1, num_channel_output=1,
                 img_rows=128, img_cols=128,
                 y=np.array([-1, 1]),  # change to output_range in the future
                 output_range=None,
                 lr_init=None, loss_function=mixedLoss(),
                 metrics_monitor=[mean_absolute_error, mean_squared_error],
                 num_poolings=4, num_conv_per_pooling=3, num_channel_first=64,weight_decay = 1E-4,
                 with_bn=True,  # don't use for F16 now
                 with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                 with_multi_lateral_filter=False,  # guilded filter bank
                 # filter parameters
                 img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                 # new settings
                 activation_conv='relu',  # options: 'elu', 'selu'
                 activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                 kernel_initializer='zeros',  # options: 'he_normal'
                 verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_dim_ordering() == "th" else -1

            x = BatchNormalization(axis=-1)(x)
            return Activation(activation_conv)(x)
    else:
        def lambda_bn(x):
            return Activation(activation_conv)(x)



    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'
        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"
        x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
        x = lambda_bn(x)

        left = Convolution2D(expand, (1, 1), padding='valid')(x)
        left = lambda_bn(left)

        right = Convolution2D(expand, (3, 3), padding='same')(x)
        right = lambda_bn(right)

        bypass = Convolution2D(2*expand, (1, 1), padding='same')(x)
        x = concatenate([left, right], axis=3)
        x = keras_add([x,bypass])
        # x = lambda_bn(x)
        return x


    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)

    # step1
    '''
    Modification descriptioin (Charles 11/16/18)
    
    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    Added Dense connections in the concatenation part of the decoding side of the Unet
        See the additional nested for loop in the decoding side of the Unet
    '''
    '''
    Below was modified by Charles
    '''
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        # conv1 = Conv2D(num_channel_first, (3, 3),
        #                padding="same",
        #                activation=activation_conv,
        #                kernel_initializer=kernel_initializer)(conv1)
        conv1 = fire_module(conv1, fire_id=i+1, squeeze=int(num_channel_first/4), expand=num_channel_first)

        conv1 = lambda_bn(conv1)
        # if (i + 1) % 2 == 0 and i != 1:
        #     conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (1 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            # conv_encoder = Conv2D(
            #     num_channel, (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_encoder)
            conv_encoder = fire_module(conv_encoder, fire_id=i+1+num_conv_per_pooling, squeeze=int(num_channel/4), expand=num_channel)

            conv_encoder = lambda_bn(conv_encoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    # conv_encoders.append(pool_encoders[-1]) #for dense implementation
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(pool_encoders[-1])
    conv_center = fire_module(pool_encoders[-1], fire_id=i+1+num_poolings*num_conv_per_pooling, squeeze=int(list_num_features[-1]/2), expand=list_num_features[-1]*2)
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(conv_center)
    conv_center = fire_module(conv_center, fire_id=i+1+num_poolings*num_conv_per_pooling, squeeze=int(list_num_features[-1]/2), expand=list_num_features[-1]*2)

    #    conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    upsample_decoders = []
    middle_decoders = []
    for i in range(1, num_poolings + 1):
        ## concate from encoding layers
        # upsample_decoder = UpSampling2D(size=(2, 2))(conv_decoders[-1])


        # # Nested U net implementation
        # upsample_decoder = Conv2DTranspose(
        #     list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
        #     activation=activation_conv,
        #     kernel_initializer=kernel_initializer)(conv_decoders[-1])
        # upsample_decoders.append(upsample_decoder)
        # if i >= 2:
        #     middle_decoder = Conv2DTranspose(
        #         list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
        #         activation=activation_conv,
        #         kernel_initializer=kernel_initializer)(conv_encoders[-i])
        #     middle_decoders.append(middle_decoder)
        #     upsample_decoder = concatenate([upsample_decoder, middle_decoders[-1]])
        #     if i >= 3:
        #         for iter in range(1,i-1):
        #             new_middle_decoder = Conv2DTranspose(
        #             list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
        #             activation=activation_conv,
        #             kernel_initializer=kernel_initializer)(middle_decoders[-i+1])
        #             middle_decoders.append(new_middle_decoder)
        #             upsample_decoder = concatenate([upsample_decoder, middle_decoders[-1]])
        #
        # upsample_decoder = concatenate(
        #         [upsample_decoder, conv_encoders[-i-1]])

        #Regular U net Implementation
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i]])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            # conv_decoder = Conv2D(
            #     list_num_features[-i], (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_decoder)

            conv_decoder = fire_module(conv_decoder, fire_id=i + 1 + num_poolings * num_conv_per_pooling,
                                      squeeze=int(list_num_features[-i]/4), expand=list_num_features[-i])


            # conv_decoder = lambda_bn(conv_decoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model



def denseUNet(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128, img_z=128,
                   y=np.array([-1, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function=mixedLoss(),
                   metrics_monitor=[mean_absolute_error, mean_squared_error],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,nb_layers_per_block=24,growth_rate=2,
                   bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                   with_bn=True,  # don't use for F16 now
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   with_multi_lateral_filter=False,  # guilded filter bank
                   # filter parameters
                   img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                   # new settings
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    else:
        def lambda_bn(x):
            return x

    nb_dense_block = num_conv_per_pooling
    def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
        ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = lambda_bn(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = lambda_bn(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                      grow_nb_filters=True, return_concat_list=False):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns: keras tensor with nb_layers of conv_block appended
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter



    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)
    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # step1
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        # conv1 = Conv2D(num_channel_first, (3, 3),
        #                padding="same",
        #                activation=activation_conv,
        #                kernel_initializer=kernel_initializer)(conv1)

        conv1, num_channel_first = __dense_block(conv1, nb_layers[i], num_channel_first, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # conv1 = lambda_bn(conv1)
        # if (i + 1) % 2 == 0 and i != 1:
        #     conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            # conv_encoder = Conv2D(
            #     num_channel, (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_encoder)

            conv_encoder, junk = __dense_block(conv_encoder, nb_layers[j], num_channel, growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_encoder = lambda_bn(conv_encoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        # pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(pool_encoders[-1])
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(conv_center)

    conv_center, junk = __dense_block(pool_encoders[-1], nb_layers[j+1], list_num_features[-1] * 2, growth_rate,
                                       bottleneck=bottleneck,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

    # conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):
        # concate from encoding layers
        #        upsample_decoder = concatenate(
        #            [UpSampling2D(size=(2, 2))(conv_decoders[-1]), conv_encoders[-i]])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i]])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            # conv_decoder = Conv2D(
            #     list_num_features[-i], (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_decoder)


            conv_decoder, junk = __dense_block(conv_decoder, nb_layers[j], list_num_features[-i], growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_decoder = lambda_bn(conv_decoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        # conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model


def denseCDUNet(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128,
                   y=np.array([-1, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function=mixedLoss(),
                   metrics_monitor=[mean_absolute_error, mean_squared_error],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,nb_layers_per_block=12,growth_rate=12,
                   bottleneck=False, dropout_rate=0.2, weight_decay=1e-4,
                   with_bn=True,  # don't use for F16 now
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   with_multi_lateral_filter=False,  # guilded filter bank
                   # filter parameters
                   img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                   # new settings
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = BatchNormalization(momentum = 0.9,axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    else:
        def lambda_bn(x):
            return x

    nb_dense_block = num_conv_per_pooling
    def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
        ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = lambda_bn(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = lambda_bn(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def __deconv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
        ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = lambda_bn(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Conv2DTranspose(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = lambda_bn(x)
            x = Activation('relu')(x)

        x = Conv2DTranspose(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x
    def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                      grow_nb_filters=True, return_concat_list=False):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns: keras tensor with nb_layers of conv_block appended
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


    def __dedense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                      grow_nb_filters=True, return_concat_list=False):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns: keras tensor with nb_layers of conv_block appended
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __deconv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter

    # layers For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)
    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # step1
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        # conv1 = Conv2D(num_channel_first, (3, 3),
        #                padding="same",
        #                activation=activation_conv,
        #                kernel_initializer=kernel_initializer)(conv1)

        conv1, num_channel_first = __dense_block(conv1, nb_layers[i], num_channel_first, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # conv1 = lambda_bn(conv1)
        # if (i + 1) % 2 == 0 and i != 1:
        #     conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            # conv_encoder = Conv2D(
            #     num_channel, (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_encoder)

            conv_encoder, junk = __dense_block(conv_encoder, nb_layers[j], num_channel, growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_encoder = lambda_bn(conv_encoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
        # pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(pool_encoders[-1])
    # conv_center = Conv2D(list_num_features[-1] * 2, (3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(conv_center)

    conv_center, junk = __dense_block(pool_encoders[-1], nb_layers[j+1], list_num_features[-1] * 2, growth_rate,
                                       bottleneck=bottleneck,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

    # conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):
        # concate from encoding layers
        #        upsample_decoder = concatenate(
        #            [UpSampling2D(size=(2, 2))(conv_decoders[-1]), conv_encoders[-i]])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2, 2), strides=(2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i]])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            # conv_decoder = Conv2D(
            #     list_num_features[-i], (3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_decoder)


            conv_decoder, junk = __dedense_block(conv_decoder, nb_layers[j], list_num_features[-i], growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_decoder = lambda_bn(conv_decoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        # conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model


def denseUNet_3D(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128, img_z=128,
                   y=np.array([-1, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function=mixedLoss(),
                   metrics_monitor=[mean_absolute_error, mean_squared_error],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=8,nb_layers_per_block=2,growth_rate=12,
                   bottleneck=False, dropout_rate=0.2, weight_decay=1e-4,
                   with_bn=True,  # don't use for F16 now
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   with_multi_lateral_filter=False,  # guilded filter bank
                   # filter parameters
                   img_sigmas=[100, 10, 1], gimg_sigmas=[1, 10, 100], spatial_sigmas=[10, 5, 1],
                   # new settings
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1, num_gpus = 1):
    # BatchNorm
    if with_bn:
        def lambda_bn(x):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            return x
    else:
        def lambda_bn(x):
            return x

    nb_dense_block = num_conv_per_pooling
    def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
        ''' Apply BatchNorm, Relu, 3x3x3 Conv3D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = lambda_bn(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = lambda_bn(x)
            x = Activation('relu')(x)

        x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                      grow_nb_filters=True, return_concat_list=False):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns: keras tensor with nb_layers of conv_block appended
        '''
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter



    # layers For 3D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
    inputs = Input((img_rows, img_cols, img_z, num_channel_input))
    if verbose:
        print('inputs:', inputs)

    # add filtered output
    if with_multi_lateral_filter:
        img = inputs[..., :num_channel_input // 2]
        gimg = inputs[..., num_channel_input // 2:]
        filtered_inputs = [inputs]
        for img_sigma, gimg_sigma, spatial_sigma in zip(img_sigmas,
                                                        gimg_sigmas,
                                                        spatial_sigmas):
            filtered_inputs.append(multi_lateral_filter(img, gimg,
                                                        img_sigma,
                                                        gimg_sigma,
                                                        spatial_sigma))
        filtered_inputs = tf.concat(filtered_inputs, axis=-1)
    else:
        filtered_inputs = inputs

    if verbose:
        print('filtered_inputs:', filtered_inputs)
    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # step1
    conv1 = inputs
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        # conv1 = Conv3D(num_channel_first, (3, 3, 3),
        #                padding="same",
        #                activation=activation_conv,
        #                kernel_initializer=kernel_initializer)(conv1)

        conv1, num_channel_first = __dense_block(conv1, nb_layers[i], num_channel_first, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # conv1 = lambda_bn(conv1)
        # if (i + 1) % 2 == 0 and i != 1:
        #     conv1 = keras_add([conv_identity[-1], conv1])
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = SpatialDropout2D(0.5)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs, conv1]
    pool_encoders = [inputs, pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            # conv_encoder = Conv3D(
            #     num_channel, (3, 3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_encoder)

            conv_encoder, junk = __dense_block(conv_encoder, nb_layers[j], num_channel, growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_encoder = lambda_bn(conv_encoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_encoder = keras_add([conv_identity[-1], conv_encoder])
        pool_encoder = MaxPooling3D(pool_size=(2, 2, 2))(conv_encoder)
        # pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    # center connection
    #    pool_encoder = SpatialDropout2D(0.5)(pool_encoder)
    #    pool_encoders[-1] = pool_encoder
    # conv_center = Conv3D(list_num_features[-1] * 2, (3, 3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(pool_encoders[-1])
    # conv_center = Conv3D(list_num_features[-1] * 2, (3, 3, 3), padding="same", activation="relu",
    #                      kernel_initializer=kernel_initializer,
    #                      bias_initializer='zeros')(conv_center)

    conv_center, junk = __dense_block(pool_encoders[-1], nb_layers[j+1], list_num_features[-1] * 2, growth_rate,
                                       bottleneck=bottleneck,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

    # conv_center = SpatialDropout2D(0.5)(conv_center)
    # conv_center = keras_add([pool_encoders[-1], conv_center])
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):
        # concate from encoding layers
        #        upsample_decoder = concatenate(
        #            [UpSampling2D(size=(2, 2))(conv_decoders[-1]), conv_encoders[-i]])
        upsample_decoder = concatenate(
            [Conv3DTranspose(
                list_num_features[-i], (2, 2, 2), strides=(2, 2, 2), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i]])

        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            # conv_decoder = Conv3D(
            #     list_num_features[-i], (3, 3, 3), padding="same",
            #     activation=activation_conv,
            #     kernel_initializer=kernel_initializer)(conv_decoder)


            conv_decoder, junk = __dense_block(conv_decoder, nb_layers[j], list_num_features[-i], growth_rate,
                                                 bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # conv_decoder = lambda_bn(conv_decoder)
            # if (j + 1) % 2 == 0 and j != 1:
            #     conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        # conv_decoder = SpatialDropout2D(0.5)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        # conv_decoder = Conv3D(num_channel_output, (1, 1, 1), padding="same", activation='linear')(conv_decoder)
        # conv_decoder = keras_add([conv_decoder, inputs[:,:,:,0:1]])
        # Add()([conv_decoder, inputs[:,:,:,0:1]]])
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv3D(num_channel_output, (1, 1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])

    # construct model
    if num_gpus>1:
        with tf.device('/cpu:0'):
            model = Model(outputs=conv_output, inputs=inputs)
    else:
        model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=lr_init)
    else:
        optimizer = Adam()
    parallel_model = None
    if num_gpus>1:
        model = multi_gpu_model(model, gpus=num_gpus)

    # print('Number of Layers: ', len(temp_model.layers))
    # print('Number of parameters: ', temp_model.count_params())

    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)
    return model