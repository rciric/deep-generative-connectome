import os
import random
from shutil import copyfile

def split_data(dir_preprocessing_list = ['/home/huang/decathlon_data/Task02_Heart_preprocessed_512_512_x_norm'],
               output_path='/home/huang/decathlon_data'):
    for dir_preprocessing in dir_preprocessing_list:

        all_file_list = os.listdir(dir_preprocessing)
        filenames = [f for f in all_file_list if f.endswith("_input.nii.gz")]
        # filenames = ['img_000.jpg', 'img_001.jpg', ...]
        filenames.sort()  # make sure that the filenames have a fixed order before shuffling
        random.seed(230)
        random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

        split_perc = 0.9
        split_1 = int(split_perc * len(filenames))
        split_2 = int(((1-split_perc)/2+split_perc) * len(filenames))
        train_filenames = filenames[:split_1]
        dev_filenames = filenames[split_1:split_2]
        test_filenames = filenames[split_2:]

        dir_train = os.path.join(output_path,'training_data')
        dir_val = os.path.join(output_path,'val_data')
        dir_test = os.path.join(output_path,'test_data')

        if not os.path.exists(dir_train):
            os.mkdir(dir_train)
        if not os.path.exists(dir_val):
            os.mkdir(dir_val)
        if not os.path.exists(dir_test):
            os.mkdir(dir_test)

        for file in train_filenames:
            copyfile(os.path.join(dir_preprocessing,file),os.path.join(dir_train,file))
            copyfile(os.path.join(dir_preprocessing,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'),os.path.join(dir_train,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'))


        for file in dev_filenames:
            copyfile(os.path.join(dir_preprocessing,file),os.path.join(dir_val,file))
            copyfile(os.path.join(dir_preprocessing,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'),os.path.join(dir_val,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'))

        for file in test_filenames:
            copyfile(os.path.join(dir_preprocessing,file),os.path.join(dir_test,file))
            copyfile(os.path.join(dir_preprocessing,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'),os.path.join(dir_test,file[:-len('_input.nii.gz')] + '_segmentation.nii.gz'))
