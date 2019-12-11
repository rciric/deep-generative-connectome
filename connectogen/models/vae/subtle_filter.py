#!/usr/bin/env python
'''
Bilateral denoising using tensorflow
'''

import numpy as np
import tensorflow as tf


def shift_image(img, i, j):
    '''
    Circular shift image by i and j

    Params
    ------
    img - Tensor, image with shape = [batch_size, nx, ny, nc]
    i - x shift
    j - y shift
    '''
    img = tf.concat([img[:, :, -j:, :], img[:, :, :-j, :]], axis=2)
    img = tf.concat([img[:, -i:, :, :], img[:, :-i, :, :]], axis=1)

    return img


def multi_lateral_filter(img, gimg, img_sigma, gimg_sigma, spatial_sigma, radius=5):
    '''
    Multi-lateral filter

    Uses the guide image to help compute filter weights to filter image
    
    Params
    ------
    img - Tensor, image with shape [batch_size, nx, ny, nc]
    gimg - Tensor, guide image with same shape as image
    radius - integer, radius of local neighborhood
    img_sigma - float, standard deviation of intensity Gaussian filter for image
    gimg_sigma - float, standard deviation of intensity Gaussian filter for guided image
    spatial_sigma - float, standard deviation of spatial Gaussian filter
    radius - int, [optional], filter radius, default=5

    Returns
    -------
    output - Tensor, filtered image with same shape as image
    '''
    
    output = 0
    weights_sum = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):

            img_ij = shift_image(img, i, j)
            gimg_ij = shift_image(gimg, i, j)
            
            weights = np.exp(-(i**2 + j**2) / spatial_sigma**2)
            weights *= tf.exp(-(img - img_ij)**2 / img_sigma**2)
            weights *= tf.exp(-(gimg - gimg_ij)**2 / gimg_sigma**2)
            
            output += weights * img_ij
            weights_sum += weights

    output /= weights_sum

    return output

