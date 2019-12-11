#!/usr/bin/env python
'''
file i/o modules and functions
'''

import pydicom
import nibabel as nib
import numpy as np
import scipy.io as sio
import os
import h5py
import logging

from subtle_utils import augment_data
#from scipy.misc import imresize
from scipy.ndimage import zoom


'''
load data with specific format
'''
def load_h5(path, key_h5='init', transpose_dims=[2, 0, 1, 3]):
    '''Loads h5 file and convert to a standard format.

    Parameters
    ----------
    path: Path to h5 file.

    Return
    ------
    numpy array.
    '''
    with h5py.File(path, 'r') as f:
        data = np.array(f[key_h5])

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)

    return data

def load_nib(path, transpose_dims=None, return_affine = False):
    '''Loads h5 file and convert to a standard format.

    Parameters
    ----------
    path: Path to nifti file.

    Return
    ------
    numpy array.
    '''
    img = nib.load(path)
    affine = None
    if return_affine == True:
        affine = img.affine
    data = img.get_data()
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)
    return data, affine

'''
get data from certain extension, here we use np or h5 format
'''
def get_data_from_ext(filepath_load, ext_data, return_data=False, return_mean=True):
    data_load = None
    value_mean = None
    value_var = None
    affine = None
    if ext_data.startswith('np'):
        data_load = np.load(filepath_load)
        # data_load = np.maximum(0, np.nan_to_num(data_load, 0))
        if return_mean:
            value_mean = np.mean(data_load)
        if return_data:
            data_load = np.array(data_load)
    elif ext_data == 'h5' or ext_data == 'hdf5':
        data_load = load_h5(filepath_load, key_h5='init',
                            transpose_dims=None)
        # data_load = np.maximum(0, np.nan_to_num(data_load, 0))
        # value_mean = np.mean(np.abs(data_load))
    elif ext_data.find('nii')>=0:
        data_load,affine = load_nib(filepath_load,return_affine=True)
        # data_load = np.maximum(0, np.nan_to_num(data_load, 0))
        value_mean = np.mean(data_load)
        # print(np.mean(np.abs(data_load)))
        value_var = np.var(data_load)
    else:
        print('not valid extension:'+ext)
    data_shape = data_load.shape
    return data_load, data_shape, affine, value_mean, value_var

'''
export file
'''

def export_to_h5(data_export, path_export, key_h5='init', dtype=np.float32, verbose=0):
    '''Export numpy array to h5.

    Parameters
    ----------
    data_export (numpy array): data to export.
    path_export (str): path to h5 file.
    key_h5: key for h5 file.
    '''
    with h5py.File(path_export,'w') as f:
        f.create_dataset(key_h5, data=data_export.astype(dtype))

    logger = logging.getLogger(__name__)
    logger.debug('H5 exported to: {}'.format(path_export))


def export_to_h5(data_export, path_export, key_h5='init', verbose=0):
    with h5py.File(path_export, 'w') as f1:
        f1.create_dataset(key_h5, data=data_export.astype(np.float32))
    if verbose:
        print('updated H5 exported to:', path_export)

'''
process file
'''
def data_reshape(data_load, shape_zoom, axis_slice=3, zoom_order=1):
    data_load_reshape = np.zeros(shape_zoom+[data_load.shape[axis_slice]])
    zoom_factor = [shape_zoom[0]/(data_load.shape[0]+0.0), shape_zoom[1]/(data_load.shape[1]+0.0), shape_zoom[2]/(data_load.shape[2]+0.0)]
    for i in range(data_load.shape[-1]):
        data_load_reshape[:,:,:,i] = zoom(np.squeeze(data_load[:,:,:,i]), zoom_factor, order=zoom_order)
    return data_load_reshape


def data_reshape_3d(data_load, shape_zoom, axis_slice=3, zoom_order=1):
    if len(data_load.shape)<4:
        data_load = data_load.reshape(data_load.shape+(1,))
    data_load_reshape = np.zeros(shape_zoom+[data_load.shape[axis_slice]])
    zoom_factor = [shape_zoom[0]/(data_load.shape[0]+0.0), shape_zoom[1]/(data_load.shape[1]+0.0), shape_zoom[2]/(data_load.shape[2]+0.0)]
    for i in range(data_load.shape[-1]):
        data_load_reshape[:,:,:,i] = zoom(np.squeeze(data_load[:,:,:,i]), zoom_factor, order=zoom_order)
    return data_load_reshape

# 2.5D augment
def mirror_roll(data_load, i_augment):
    nx,ny,nz,nc = data_load.shape
    data_load2 = np.zeros([nx+abs(i_augment)*2,ny,nz,nc])
    data_load2[abs(i_augment):-abs(i_augment),:,:,:] = data_load
    for i in range(0,abs(i_augment)):
        data_load2[i,:,:,:]=data_load[0,:,:,:]
    for i in range(0,abs(i_augment)):
        data_load2[nx+i+abs(i_augment),:,:,:]=data_load[-1,:,:,:]
    data_load_shift = np.roll(np.array(data_load2), i_augment, axis=0)
    return data_load_shift[abs(i_augment):-abs(i_augment),:,:,:]



def preprocess_data(data, list_augments=[], scale_by_mean=True, slices=None,
                    scale_factor=-1, augment_25D=0, channel_as_contrast=0, shape_resize=None, zoom_order=3):

    # scale
    data = np.copy(data)
    #data *= data > 0.0  # mask out negative values
    if scale_by_mean:
        data /= np.mean(np.abs(data))
        print('normalized by mean of abs')
    data *= data > 0.0
    if scale_factor >=0:
        data *= scale_factor

    # resize
    if shape_resize:
        data_resize = np.zeros(
            [data.shape[0], shape_resize[0], shape_resize[1], data.shape[-1]])
        zoom_factor = [
            shape_resize[0]/(data.shape[1]+0.0), shape_resize[1]/(data.shape[2]+0.0)]
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                data_resize[i, :, :, j] = zoom(
                    np.squeeze(data[i, :, :, j]), zoom_factor, order=zoom_order)
        data = data_resize

    # augment 2.5D
    if augment_25D == 0:
        data = data[:, :, :, :, np.newaxis]
    else:
        list_shifts = []
        for i_augment in range(1, augment_25D+1):
            data_shift = np.array(data)
            data_shift = mirror_roll(np.array(data), -i_augment)
            # data_shift = np.roll(np.array(data), -i_augment, axis=0)
            data_shift = data_shift[:, :, :, :, np.newaxis]
            list_shifts.append(data_shift)
        list_shifts.append(np.array(data)[:, :, :, :, np.newaxis])
        for i_augment in range(1, augment_25D+1):
            data_shift = np.array(data)
            data_shift = mirror_roll(np.array(data), +i_augment)
            # data_shift = np.roll(np.array(data), +i_augment, axis=0)
            data_shift = data_shift[:, :, :, :, np.newaxis]
            list_shifts.append(data_shift)
        data = np.concatenate(list_shifts, axis=-1)

    # extract slices
    if slices is not None:
        data_subslice = data[np.array(slices), :, :, :, :]
        if np.ndim(data_subslice) < np.ndim(data):
            data_subslice = data_subslice[np.newaxis, :, :, :, :]
        data = data_subslice

    # augmentation
    if len(list_augments) > 0:
        list_data = []
        for augment in list_augments:
            data_augmented = augment_data(
                data, axis_xy=[1, 2], augment=augment)
            list_data.append(data_augmented.reshape(data.shape))
        data = np.concatenate(list_data, axis=0)

    # transpose
    if channel_as_contrast:
        data = np.reshape(data, [data.shape[0], data.shape[1],
                                           data.shape[2],
                                           data.shape[3] * data.shape[4]])
    else:
        data = np.transpose(data, [0, 3, 1, 2, 4])
        data = np.reshape(data, [data.shape[0] * data.shape[1],
                                           data.shape[2], data.shape[3],
                                           data.shape[4]])
    return data

def resize_data(data, shape_resize=None, zoom_order=3):
    '''Resize data.

    Parameters
    ----------
    data: numpy array, input data of shape [slices, y, x, channel].
    shape_resize: None or length 2 tuple, shape to be resized to.

    Returns
    -------
    data_resize: numpy array.
    '''
    if shape_resize:
        data_resize = np.zeros([data.shape[0], shape_resize[0], shape_resize[1], data.shape[-1]])
        zoom_factor = [shape_resize[0] / data.shape[1], shape_resize[1] / data.shape[2]]
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                data_resize[i, :, :, j] = zoom(data[i, :, :, j], zoom_factor, order=zoom_order)
    else:
        data_resize = data

    return data_resize
