#!/usr/bin/env python
'''
Generator used for training
'''

import numpy as np
import h5py
from scipy.ndimage import zoom
from subtle_utils import augment_data


class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, dim_x=256, dim_y=256, dim_z=5, dim_25d=-1, dim_output=1,
                 batch_size=4, shuffle=True, verbose=1,
                 scale_data=0.1, scale_baseline=1.0,
                 list_fileinfo_sample=[],
                 normalize_data=True, normalize_per_sample=False,
                 resize_data=True,
                 mask_data=True, threshold_mask=0.01,
                 sanitize_data=True,
                 augment_data=True,
                 axis_slice=0,
                 para={}):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_25d = dim_25d
        self.dim_output = dim_output
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.scale_data = scale_data
        self.scale_baseline = scale_baseline
        self.para = para
        self.list_fileinfo_sample = list_fileinfo_sample
        self.normalize_data = normalize_data
        self.normalize_per_sample = normalize_per_sample
        self.resize_data = resize_data
        self.files_h5 = {}
        self.mask_data = mask_data
        self.threshold_mask = threshold_mask
        self.sanitize_data = sanitize_data
        self.augment_data = augment_data
        self.axis_slice = axis_slice

    def logout(self, level=1, content=''):
        if level <= self.verbose:
            print(content)
        return

    # @threadsafe_generator
    def generate(self, list_fileinfo_sample=[], list_sample_total=[], ext_data='.npz'):
        'Generates batches of samples'
        if len(list_fileinfo_sample) <= 0:
            list_fileinfo_sample = self.list_fileinfo_sample
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_sample_total)
            # self.logout(level=2, content='reorder indexes:{0}'.format(indexes))
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            # self.logout(level=2, content='number of batches:{0}'.format(imax))
            for i in range(imax):
                # Find list of IDs
                list_sample_inbatch = [list_sample_total[k]
                                 for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                self.logout(
                    level=3, content='index in current batch:{0}'.format(list_sample_inbatch))
                # Generate data
                X, Y, W = self.__data_generation(
                    list_fileinfo_sample, list_sample_inbatch, ext_data)
                if (np.max(X.flatten()) <= 0 and np.max(Y.flatten()) <= 0):
                    continue
                self.logout(
                    level=2, content='generated dataset size:{0},{1}'.format(X.shape, Y.shape))
                yield X, Y, W

    def __get_exploration_order(self, list_sample_total):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_sample_total))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __get_data_from_h5(self, filepath_load, index_slice=-1, axis_slice=-1, augment_25d=1):
        'Generates 2.5d slices from filepath and index of slice'
        with h5py.File(filepath_load, 'r') as f:
            keys = list(f.keys())
            key_data = keys[0]
            # get data
            data_load = f[key_data]
            # extract
            augment_slice_expand = int((augment_25d-1)/2)
            self.logout(level=4,
                        content='25d sample:{0},{1}'.format(index_slice, augment_slice_expand))
            if axis_slice == -1:
                data_extract = data_load[:, :, (index_slice-augment_slice_expand):(
                    index_slice+augment_slice_expand+1)]
                data_load = np.array(data_extract)
            if axis_slice == 0:
                data_extract = data_load[(
                    index_slice-augment_slice_expand):(index_slice+augment_slice_expand+1)]
                data_load = np.array(data_extract)
                if np.ndim(data_load)>3:
                    data_load = np.transpose(data_load, [1, 2, 0, -1])
                    data_load = np.reshape(data_load, [data_load.shape[0], data_load.shape[1], data_load.shape[2]*data_load.shape[3]])
                else:
                    data_load = np.transpose(data_load, [1, 2, 0])

        return data_load

    def __data_reshape(self, data_load, shape_zoom):
        'Reshape data to certain dimension, deprecated later'
        data_load_reshape = np.zeros(shape_zoom+[data_load.shape[-1]])
        zoom_factor = [
            shape_zoom[0]/(data_load.shape[0]+0.0), shape_zoom[1]/(data_load.shape[1]+0.0)]
        for i in range(data_load.shape[-1]):
            data_load_reshape[:, :, i] = zoom(
                np.squeeze(data_load[:, :, i]), zoom_factor)
        return data_load_reshape

    def __data_generation(self, list_filepath_sample, list_sample_inbatch, ext_data='h5'):
        # X : (n_samples, v_size, v_size, v_size, n_channels)
        'Generates data of batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim_x,
                      self.dim_y, self.dim_z))
        Y = np.empty((self.batch_size, self.dim_x,
                      self.dim_y, self.dim_output))
        if 'index_mid' in self.para and self.para['index_mid'] > 0:
            index_mid = int(self.para['index_mid'])
        else:
            index_mid = int((self.dim_25d-1)/2)

        # generate data
        Weights = []
        for index_in_batch, sample_content in enumerate(list_sample_inbatch):
            # get input and output
            if type(sample_content)==list:
                filepaths_input, filepath_output, index_slice, sample_weight = sample_content
            elif  type(sample_content)==dict:
                filepaths_input = sample_content['filepath_input']
                filepath_output = sample_content['filepath_output']
                index_slice = sample_content['index_slice']
                sample_weight = sample_content['sample_weight']
                
            # load inputs
            num_input_modality = len(filepaths_input)
            data_inputs = []
            for index_input in range(num_input_modality):
                data_inputs.append(self.__get_data_from_h5(filepaths_input[index_input],
                                                           index_slice=index_slice,
                                                           axis_slice=self.axis_slice,
                                                           augment_25d=self.dim_25d))

            data_full = self.__get_data_from_h5(filepath_output, 
                                                index_slice=index_slice,
                                                axis_slice=self.axis_slice, 
                                                augment_25d=self.dim_25d)

            # # # scale using saved scale
            if self.normalize_data:
                if not self.normalize_per_sample:
                    for index_input in range(num_input_modality):
                        data_inputs[index_input] /= scales_mean[index_input]

                    data_full /= scales_mean[-1]
                else:
                    for index_input in range(num_input_modality):
                        data_inputs[index_input] /= np.mean(data_inputs[index_input]) + 0.0
                    data_full /= np.mean(data_full) + 0.0

            # reshape
            if self.resize_data:
                dim_reshape = [self.dim_x, self.dim_y]
                for index_input in range(len(data_inputs)):
                    data_inputs[index_input] = self.__data_reshape(
                        data_inputs[index_input], dim_reshape)

                data_full = self.__data_reshape(data_full, dim_reshape)

            # mask
            if self.mask_data:
                data_mask = data_full > self.threshold_mask
                for index_input in range(len(data_inputs)):
                    data_inputs[index_input] *= data_mask
                data_full *= data_mask

            # concatenate
            X[index_in_batch, :, :, :] = np.concatenate(data_inputs, axis=-1)
            # self.logout(level=4, content='X dim:{0}'.format(X.shape))
            if data_full.shape[-1] > 1:
                Y[index_in_batch, :, :, :] = data_full[:, :, index_mid:(index_mid+1)]
            else:
                Y[index_in_batch, :, :, :] = data_full[:, :, :]
            # self.logout(level=4, content='Y dim:{0}'.format(Y.shape))
            Weights.append(sample_weight)

        # scale
        if self.scale_data or np.abs(self.scale_data) > 0:
            X = X * self.scale_data
            Y = Y * self.scale_data

        # # sanitize data
        if self.sanitize_data:
            X = np.maximum(0, np.nan_to_num(X, 0))
            Y = np.maximum(0, np.nan_to_num(Y, 0))

        # residual for train based on average image
        if self.scale_baseline or np.abs(self.scale_baseline) > 0:
            X_baseline = X[:, :, :, index_mid:(index_mid+1)]
            Y = Y - self.scale_baseline * X_baseline

        # clip and mask
        if 'clip_output' in self.para:
            Y = np.maximum(np.minimum(Y, self.para['clip_output'][1]),
                           self.para['clip_output'][0])

        # random rotate/transform/augmentation
        if self.augment_data and 'augmentation' in self.para:
            flipxy = np.random.randint(self.para['augmentation'][0])
            flipx = np.random.randint(self.para['augmentation'][1])
            flipy = np.random.randint(self.para['augmentation'][2])
            flipc = np.random.randint(self.para['augmentation'][-1])
            flipc_segment = num_input_modality
            augment = {'flipxy': flipxy, 'flipx': flipx,
                       'flipy': flipy, 'flipc': flipc, 
                       'flipc_segment': flipc_segment}
            X = augment_data(X, axis_xy=[1, 2], augment=augment)
            Y = augment_data(Y, axis_xy=[1, 2], augment=augment)


        Weights = np.array(Weights)
        return X, Y, Weights

