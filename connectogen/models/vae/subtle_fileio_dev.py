#!/usr/bin/env python
'''
file i/o modules and functions
'''

import dicom
import nibabel as nib
import numpy as np
import scipy.io as sio
import os
import h5py
import logging

from subtle_utils import augment_data
from scipy.misc import imresize
from scipy.ndimage import zoom

'''
load data with specific format
'''
def load_h5(path, key_h5='init', transpose_dims=[2, 0, 1, 3]):
    '''Loads h5 file and convert to a standard format.
    
    Parameters
    ----------
    path: str, path to h5 file.
    key_h5: str, data key.
    transpose_dims: length-4 tuple, transpose dimensions.
    
    Return
    ------
    data: numpy array.
    '''
    with h5py.File(path, 'r') as f:
        data = np.array(f[key_h5])
        
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    if transpose_dims:
        data = np.transpose(data, transpose_dims)

    return data

def load_nib(path, transpose_dims=None):
    '''Loads h5 file and convert to a standard format.
    
    Parameters
    ----------
    path: Path to nifti file.
    
    Return
    ------
    numpy array.
    '''    
    img = nib.load(path)
    data = img.get_data()
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)
    return data

'''
get data from certain extension, here we use np or h5 format
'''
def get_data_from_ext(filepath_load, ext_data, return_data=False, return_mean=True):
    data_load = None
    value_mean = None
    # np
    if ext_data.startswith('np'):
        data_load = np.load(filepath_load)
        if return_mean: 
            value_mean = np.mean(np.abs(data_load).flatten())        
        if return_data:
            data_load = np.array(data_load)        
    # hdf5
    elif ext_data == 'h5' or ext_data == 'hdf5':
        data_load = load_h5(filepath_load, key_h5='init', 
                            transpose_dims=None)
        if return_data:
            data_load = np.array(data_load)
        if return_mean:
            value_mean = np.mean(np.abs(np.array(data_load)))

    # nifty
    elif ext_data.find('nii')>=0:
        data_load = load_nib(filepath_load)
        if return_mean: 
            value_mean = np.mean(np.abs(data_load).flatten())        

    else:
        print('not valid extension:'+ext)
    data_shape = data_load.shape 
    return data_load, data_shape, value_mean 


def export_to_h5(data_export, path_export, key_h5='init', dtype=np.float32):
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


'''
process file
'''       
def data_reshape(data_load, shape_zoom, axis_slice=2, zoom_order=2):
    data_load_reshape = np.zeros(shape_zoom+[data_load.shape[axis_slice]])
    zoom_factor = [shape_zoom[0]/(data_load.shape[0]+0.0), shape_zoom[1]/(data_load.shape[1]+0.0)]
    for i in range(data_load.shape[-1]):
        data_load_reshape[:,:,i] = zoom(
            np.squeeze(data_load[:,:,i]), 
            zoom_factor,
            order=zoom_order)
    return data_load_reshape


def mirror_roll(data, i_augment=0, copy_end_slices=True):
    # return identify 
    if i_augment == 0:
        return np.array(data) 
    if copy_end_slices:   
        nx, ny, nz, nc = data.shape
        data2 = np.zeros([nx + abs(i_augment) * 2, ny, nz, nc])
        data2[abs(i_augment):-abs(i_augment), :, :, :] = data
        for i in range(abs(i_augment)):    
            data2[i, :, :, :] = data[0, :, :, :]
        for i in range(abs(i_augment)):    
            data2[nx + i + abs(i_augment), :, :, :] = data[-1, :, :, :]        
        data_shift = np.roll(np.array(data2),  i_augment,  axis=0)
        return data_shift[abs(i_augment):-abs(i_augment), :, :, :]
    else:
        print('faster just use np.roll')
        return np.roll(np.array(data),  i_augment,  axis=0)


def scale_data(data, mask_negative=True, scale_by_mean=True, scale_factor=1):
    '''Scale data and mask out negative values.

    Parameters
    ----------
    data: numpy array, input data.
    scale_by_mean: bool, toggle normalize by mean.
    scale_factor: float, scale factor.

    Returns
    -------
    data_scale: numpy array.
    '''
    data_scale = data.copy()
    
    if mask_negative:
        data_scale *= data > 0.0
    
    if scale_by_mean:
        data_scale /= np.mean(data_scale)
    
    if scale_factor>=0:
        data_scale *= scale_factor

    return data_scale


def resize_data(data, shape_resize=None, zoom_order=2):
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


def augment_25D_data(data, augment_25D=0, copy_end_slices=True):
    '''Augment data 2.5D.
    
    Parameters
    ----------
    data: numpy array, input data of shape [slices, y, x, channel].
    augment_25D: int, number of augmentations.

    Returns
    -------
    data_augment_25D: numpy array of shape [slices, y, x, channel, augments]
    '''
    augments = [mirror_roll(data, shift, copy_end_slices) for shift in range(-augment_25D, augment_25D + 1)]
    data_augment_25D = np.stack(augments, axis=-1)
    return data_augment_25D


def extract_slices(data, slices=None):
    '''Extract slices.
    
    Parameters
    ----------
    data: numpy array, input data of shape [slices, y, x, channel, augments].
    slices: None or list of ints, slice indices.

    Returns
    -------
    data_subslice: numpy array of shape [slices, y, x, channel, augments]
    '''    
    if slices is None:
        return data
    else:
        data_subslice = data[np.array(slices)]
        if data_subslice.ndim < data.ndim:
            data_subslice = np.expand_dims(data_subslice, axis=0)
            
        return data_subslice


def preprocess_data(data, list_augments=[], 
                    mask_negative=True, scale_by_mean=True, scale_factor=1, slices=None,
                    axis_xy=[1, 2], augment_25D=0, channel_as_contrast=True, shape_resize=None, zoom_order=2, copy_end_slices=True):

    # process data
    data = scale_data(data, mask_negative=mask_negative, 
                      scale_by_mean=scale_by_mean, scale_factor=scale_factor)

    data = resize_data(data, shape_resize=shape_resize, zoom_order=zoom_order)

    data = augment_25D_data(data, augment_25D=augment_25D, copy_end_slices=copy_end_slices)

    data = extract_slices(data, slices=slices)

    # augmentation
    if len(list_augments) > 0:
        list_data = []
        for augment in list_augments:
            data_augmented = augment_data(data, axis_xy=[1, 2], augment=augment)
            list_data.append(data_augmented.reshape(data.shape))
            
        data = np.concatenate(list_data, axis=0)

    # transpose, channel as contrasts vs channel as samples
    if channel_as_contrast:
        data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2],
                                 data.shape[3] * data.shape[4]])
    else:
        data = np.transpose(data, [0, 3, 1, 2, 4])
        data = np.reshape(data, [data.shape[0] * data.shape[1],
                                 data.shape[2], data.shape[3], data.shape[4]])
    return data

