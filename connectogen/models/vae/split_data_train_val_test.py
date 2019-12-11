import os
import random
from shutil import copyfile
import glob
orig_dir = '/home/huang/cs236_project/data/preprocessed_connectomes/'
# dir_preprocessing_list = ['Task01_BrainTumour_preprocessed','Task02_Heart_preprocessed','Task03_Liver_preprocessed',\
# 'Task05_Prostate_preprocessed','Task06_Lung_preprocessed','Task07_Pancreas_preprocessed',\
# 'Task08_HepaticVessel_preprocessed','Task09_Spleen_preprocessed','Task10_Colon_preprocessed']

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

dir_preprocessing_list = ['derivatives']

for dir_preprocessing in dir_preprocessing_list:

    dir_preprocessing = os.path.join(orig_dir,dir_preprocessing)
    # filenames = [os.path.basename(x) for x in glob.glob(os.path.join(dir_preprocessing,'ds*/sub*_desc-schaefer100x7_connectome.1D'))]
    filenames = glob.glob(os.path.join(dir_preprocessing,'ds*/sub*_desc-schaefer100x7_connectome.1D'))
    # filenames = ['img_000.jpg', 'img_001.jpg', ...]
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    print(filenames)
    random.seed(230)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_perc = 0.9
    split_1 = int(split_perc * len(filenames))
    split_2 = int(((1-split_perc)/2+split_perc) * len(filenames))
    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    dir_train = os.path.join(orig_dir,'training_data')
    dir_val = os.path.join(orig_dir,'val_data')
    dir_test = os.path.join(orig_dir,'test_data')

    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    if not os.path.exists(dir_val):
        os.mkdir(dir_val)
    if not os.path.exists(dir_test):
        os.mkdir(dir_test)

    for file in train_filenames:
        if is_non_zero_file(file):
            basename = os.path.basename(file)
            basename = basename[:-3]+'.txt'
            copyfile(file,os.path.join(dir_train,basename))




    for file in dev_filenames:
        if is_non_zero_file(file):
            basename = os.path.basename(file)
            basename = basename[:-3]+'.txt'
            copyfile(file,os.path.join(dir_val,basename))


    for file in test_filenames:
        if is_non_zero_file(file):
            basename = os.path.basename(file)
            basename = basename[:-3]+'.txt'
            copyfile(file,os.path.join(dir_test,basename))
