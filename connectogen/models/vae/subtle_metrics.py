import tflearn
import numpy as np
from keras import backend as K
import tensorflow as tf
# use skimage metrics
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
# psnr with TF
from functools import partial
import pywt


try:
    from keras.losses import mean_absolute_error, mean_squared_error, binary_crossentropy, kullback_leibler_divergence, categorical_crossentropy
    # import keras_contrib.backend as KC
    from keras import backend as K
    from keras.layers import *
    from keras import models
    from keras.applications.vgg16 import VGG16

    from tensorflow import log as tf_log
    from tensorflow import constant as tf_constant


except:
    print('import keras and tf backend failed')


def ssim_loss(y_true, y_pred):
    kernel = [3, 3]
    k1 = 0.01
    k2 = 0.03
    kernel_size = 3
    max_value = 1.0
    cc1 = (k1 * max_value) ** 2
    cc2 = (k2 * max_value) ** 2
    y_true = KC.reshape(y_true, [-1] + list(KC.int_shape(y_pred)[1:]))
    y_pred = KC.reshape(y_pred, [-1] + list(KC.int_shape(y_pred)[1:]))

    patches_pred = KC.extract_image_patches(
        y_pred, kernel, kernel, 'valid', K.image_data_format())
    patches_true = KC.extract_image_patches(
        y_true, kernel, kernel, 'valid', K.image_data_format())

    bs, w, h, c1, c2, c3 = KC.int_shape(patches_pred)

    patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    # Get mean
    u_true = KC.mean(patches_true, axis=-1)
    u_pred = KC.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get covariance
    covar_true_pred = K.mean(
        patches_true * patches_pred, axis=-1) - u_true * u_pred

    ssim = (2 * u_true * u_pred + cc1) * (2 * covar_true_pred + cc2)
    denom = (K.square(u_true) + K.square(u_pred) + cc1) * \
        (var_pred + var_true + cc2)
    ssim /= denom

    return K.mean((1.0 - ssim) / 2.0)


def perceptual_loss(y_true, y_pred):
    '''
    Loss function to calculate 2D perceptual loss

    Parameters
    ----------
    y_ture : float
        4D true image numpy array (batches, xres, yres, channels)
    y_pred : float
        4D test image numpy array (batches, xres, yres, channels)

    Returns
    -------
    float
        RMSE between extracted perceptual features

    @author: Akshay Chaudhari <akshay@subtlemedical.com>
    Copyright Subtle Medical (https://www.subtlemedical.com)
    Created on 2018/04/20

    '''

    n_batches, xres, yres, n_channels = K.get_variable_shape(y_true)

    vgg = VGG16(include_top=False,
                weights='imagenet',
                input_shape=(xres, yres, 3))

    loss_model = models.Model(inputs=vgg.input,
                              outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    # Convert to a 3D image and then calculate the RMS Loss
    y_true_rgb = tf.image.grayscale_to_rgb(y_true/K.max(y_true), name=None)
    y_pred_rgb = tf.image.grayscale_to_rgb(y_pred/K.max(y_true), name=None)

    loss = K.mean(K.square(loss_model(y_true_rgb) - loss_model(y_pred_rgb)))

    return loss


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    try:
        # use theano
        return 20.*np.log10(K.max(y_true)) - 10. * np.log10(K.mean(K.square(y_pred - y_true)))
    except:
        denominator = tf_log(tf_constant(10.0))
        return 20.*tf_log(K.max(y_true)) / denominator - 10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator
    return 0

# segmetnation related loss
def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


# segmentation loss
def seg_crossentropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


# multi-class segmentation loss
def multi_class_seg_crossentropy(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

def roc_auc_score(y_true,y_pred):
    return tflearn.objectives.roc_auc_score(y_pred,y_true)


def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    return 2*((precision(y_true,y_pred)*recall(y_true,y_pred))/(precision(y_true,y_pred)+recall(y_true,y_pred)))

def dice_coef_multilabel(y_true, y_pred, numLabels=12):
    dice=numLabels-1
    #assuming channel last configuration
    #assuming background class is in the last channel
    for index in range(numLabels-1):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice


def weighted_categorical_crossentropy(y_true, y_pred):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    # weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

    # weights = np.array([10, 2, 1, 10, 10, 1, 1, 10, 1, 10, 10, 0])  # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    weights = np.ones((23))
    weights = K.variable(weights)

    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)

    return loss

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# 2d implementation (set (0, 1, 2) to (0, 1, 2, 3) for 3d)
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T

def tversky_loss_and_wce(y_true, y_pred):
    return tversky_loss(y_true,y_pred)+weighted_categorical_crossentropy(y_true, y_pred)

# mixed loss


def mixedLoss(weight_l1=0.5, weight_ssim=0.5, weight_perceptual_loss=0):
    if weight_perceptual_loss > 0:
        def loss_func(x, y): return mean_absolute_error(x, y)*weight_l1 + \
            ssim_loss(x, y)*weight_ssim + perceptual_loss(x, y) * \
            weight_perceptual_loss
    else:
        def loss_func(x, y): return mean_absolute_error(
            x, y)*weight_l1 + ssim_loss(x, y)*weight_ssim
    return loss_func

sd_weights = [0.5,0.5,10,1,1/3,0.1]

#def ssim_and_perc_loss(y_true,y_pred):
#    return sd_weights[0]*ssim_loss(y_true,y_pred)+sd_weights[1]*perceptual_loss(y_true,y_pred)


def sce_and_dice_loss(y_true, y_pred):
    return sd_weights[0]*seg_crossentropy(y_true,y_pred)+sd_weights[1]*dice_coef_loss(y_true,y_pred)

def sce_and_ssim_loss(y_true, y_pred):
    return sd_weights[0]*seg_crossentropy(y_true,y_pred)+sd_weights[1]*ssim_loss(y_true,y_pred)

def sce_and_ssim_with_l1_loss(y_true, y_pred):
    return sd_weights[4]*seg_crossentropy(y_true,y_pred)+sd_weights[4]*ssim_loss(y_true,y_pred)+sd_weights[4]*mean_absolute_error(y_true,y_pred)

def ssim_sce_and_dice_loss(y_true,y_pred):
    return sd_weights[2]*seg_crossentropy(y_true,y_pred)+sd_weights[3]*dice_coef_loss(y_true,y_pred) \
            + sd_weights[2]*ssim_loss(y_true,y_pred)

def psnr(im_gt, im_pred):
    return 20*np.log10(np.max(im_gt.flatten())) - 10 * np.log10(np.mean((im_pred.flatten()-im_gt.flatten())**2))

def dice_and_l2(y_true, y_pred):
    return dice_coef_loss(y_true,y_pred) + 0.2*mean_squared_error(y_true,y_pred)

# get error metrics, for psnr, ssimr, rmse, score_ismrm


def getErrorMetrics(im_pred, im_gt, mask=None):

    # flatten array
    im_pred = np.array(im_pred).astype(np.float).flatten()
    im_gt = np.array(im_gt).astype(np.float).flatten()
    if mask is not None:
        mask = np.array(mask).astype(np.float).flatten()
        im_pred = im_pred[mask > 0]
        im_gt = im_gt[mask > 0]
    mask = np.abs(im_gt.flatten()) > 0

    # check dimension
    assert(im_pred.flatten().shape == im_gt.flatten().shape)

    # NRMSE
    rmse_pred = compare_nrmse(im_gt, im_pred)

    # PSNR
    try:
        psnr_pred = compare_psnr(im_gt, im_pred)
    except:
        psnr_pred = psnr(im_gt, im_pred)
        # print('use psnr')

    # ssim
    data_range = np.max(im_gt.flatten()) - np.min(im_gt.flatten())
    ssim_pred = compare_ssim(im_gt, im_pred, data_range=data_range)
    ssim_raw = compare_ssim(im_gt, im_pred)
    score_ismrm = sum((np.abs(im_gt.flatten()-im_pred.flatten())
                       < 0.1)*mask)/(sum(mask)+0.0)*10000

    return {'rmse': rmse_pred, 'psnr': psnr_pred, 'ssim': ssim_pred,
            'ssim_raw': ssim_raw, 'score_ismrm': score_ismrm}

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.

    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.

    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?

    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # References

    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.

    https://en.wikipedia.org/wiki/Jaccard_index

    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# def grad(f, x):
#     return Lambda(lambda u: K.gradients(u[0], u[1]), output_shape=[2])([f, x])
#
# def ngrad(f, x, n):
#     if 0 == n:
#         return f
#     else:
#         return Lambda(lambda u: K.gradients(u[0], u[1]), output_shape=[2])([ngrad( f, x, n - 1 ), x])
#
#
# def sobolev_wrapper(input_tensor):
#     def sobolev_l2(y_true, y_pred):
#         """ reconstruction loss """
#         # return K.mean(K.square(y_true - y_pred)+K.square(tf.gradients(y_true,input_tensor)[0]\
#         #     - tf.gradients(y_pred,input_tensor)[0]))
#         return K.mean(K.square(y_true - y_pred)+K.square(ngrad(y_true,input_tensor) \
#                 - ngrad(y_pred,input_tensor)))
#     return sobolev_l2
