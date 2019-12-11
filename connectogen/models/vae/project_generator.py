import numpy as np
import keras
import os
import tensorflow as tf
import keras.backend as K
import nibabel as nib
import glob

from keras.preprocessing import image
from subtle_fileio import *
from sklearn.decomposition import PCA
import random

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folder_data, batch_size=1, num_contrast = 1,
                target_size=(128,128,128), shuffle=True, num_gpus = 1, verbose = 0, fp16 = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = list(target_size)[-1]
        self.folder_data = folder_data
        self.num_contrast = num_contrast
        self.num_gpus = num_gpus
        self.verbose = verbose

        self.fp16 = fp16
        all_file_list=os.listdir(folder_data)
        self.data_list = sorted([f for f in all_file_list if f.endswith(".txt")])
        self.on_epoch_end()
        # print(self.data_list)


    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size))

    # def __getitem__(self, index):
    #      # Generate indexes of the batch
    #     indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    #
    #     # Find list of IDs
    #     list_IDs_temp = [self.data_list[k] for k in indexes]
    #
    #     # Generate data
    #     X, y = self.__data_generation(list_IDs_temp)
    #     #print(list_IDs_temp)
    #     return ([X, y], [y])
    def __getitem__(self, index):
         # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.data_list[k] for k in indexes]

        # Generate data
        x = self.__data_generation(list_IDs_temp)
        outputs_list = [x,np.zeros(x.shape)]
        return ([x],outputs_list)


    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim)) # data


        #X = X_batch/255.0
        #Y_batch = np.max(Y_batch,axis=-1).reshape(Y_batch.shape[0],Y_batch.shape[1],Y_batch.shape[2],1)/255.0
        # if self.shuffle == True:
            # random.shuffle(list_IDs_temp)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img_path = os.path.join(self.folder_data, ID)

            # img = image.load_img(gt_path, target_size=self.dim)
            # y = image.img_to_array(img)
            # y = np.max(y,axis=-1).reshape(self.dim[0], self.dim[1],1)

            # Y[i,:,:] = y/255.0
            data_load = np.squeeze(np.loadtxt(img_path))
            data_load = np.nan_to_num(data_load)

            # data_load = np.maximum(0, np.nan_to_num(data_load, 0))
            if self.fp16:
                data_load = data_load.astype(np.float16)
            else:
                data_load = data_load.astype(np.float32)



            X[i,:] = data_load


        return X

    def return_data_list(self):
        unshuffled_data_list = []
        for i, ID in enumerate(self.data_list):
            unshuffled_data_list.append(ID[:-13])

        return unshuffled_data_list
    def generate(self):
        while 1:
    #         # Generate order of exploration of dataset
            indexes = self.indexes
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
            # print(indexes)
            # self.logout(level=2, content='reorder indexes:{0}'.format(indexes))
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            # self.logout(level=2, content='number of batches:{0}'.format(imax))
            for i in range(imax):
                #
                yield self.__getitem__(i)
