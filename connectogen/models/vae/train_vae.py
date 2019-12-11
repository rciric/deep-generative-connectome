"""
train atlas-based alignment with MICCAI2018 version of VoxelMorph,
specifically adding uncertainty estimation and diffeomorphic transforms.
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
from keras.preprocessing import image
from subtle_fileio import *
from subtle_metrics import *
from sklearn.decomposition import PCA
import nibabel as nib
from project_generator import DataGenerator

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.models import load_model

# project imports
import networks
import losses
from cyclic_lr import SGDRScheduler



def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    # config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = limit
    config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    set_session(tf_Session(config=config))

def train(data_dir,
          val_data_dir,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          batch_size,
          load_model_file,
          steps_per_epoch,
          initial_epoch=0):
    """
    model training function

    """
    # prior_lambda = np.sqrt(2*np.pi*image_sigma**(3/2))/6
    nb_gpus = len(gpu_id.split(','))
    # nb_gpus = 1
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)


    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    train_vol_names = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    # print(train_vol_names)
    print('Number of Training Cases: ', len(train_vol_names))

    ref_vol = np.squeeze(np.loadtxt(train_vol_names[0]))

    vol_size = ref_vol.shape
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"
    if not val_data_dir=='':
        val_vol_names = glob.glob(os.path.join(val_data_dir, '*.txt'))
        # print(val_vol_names)
        print('Number of Validation Cases: ', len(val_vol_names))
        random.shuffle(val_vol_names)

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # gpu handling
    gpu = '/gpu:%d' % 0 #gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # set_session(tf.Session(config=config))
    setKerasMemory(1)

    enc_nf = [256]
    dec_nf = [256,list(vol_size)[-1]]
    model = networks.fc_vae(vol_size, enc_nf, dec_nf, latent_dim = 128)



    # load initial weights
    if load_model_file is not None and load_model_file != "":
        model.load_weights(load_model_file)

    # save first iteration
    model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))
    '''
    original loss below
    '''
    # compile
    # note: best to supply vol_shape here than to let tf figure it out.
    flow_vol_shape = model.outputs[-1].shape[1:-1]

    reg_loss_class = losses.Losses(flow_vol_shape=flow_vol_shape)

    model_losses = [reg_loss_class.recon_loss, reg_loss_class.kl_loss] #prob gADx_vae

    loss_weights = [1, 1] #gADx_vae



    '''

    '''

    data_gen_args = dict(#featurewise_center=False,
        #featurewise_std_normalization=False,
        #rotation_range=0.0,
        #width_shift_range=0.0,
        #height_shift_range=0.0,

        shuffle=True,
        batch_size = batch_size,
        folder_data = data_dir,
        target_size = vol_size,
        num_gpus = nb_gpus,
        verbose = 0,
        fp16 = False)
    val_gen_args = dict(  # featurewise_center=False,
        # featurewise_std_normalization=False,
        # rotation_range=0.0,
        # width_shift_range=0.0,
        # height_shift_range=0.0,
        shuffle = False,
        batch_size= batch_size,
        folder_data= val_data_dir,
        target_size= vol_size,
        num_gpus = nb_gpus,
        verbose = 0,
        fp16 = False)

    training_generator = DataGenerator(**data_gen_args).generate()
    val_generator = DataGenerator(**val_gen_args).generate()
    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # multi-gpu support
    if nb_gpus > 1:
        save_callback = nrn_gen.ModelCheckpointParallel(save_file_name,
            monitor='val_loss',verbose = 1,save_best_only=True, mode = 'min')
        mg_model = multi_gpu_model(model, gpus=nb_gpus)

    # single gpu
    else:
        save_callback = ModelCheckpoint(save_file_name,
            monitor='val_loss',verbose = 1,save_best_only=True, mode = 'min')
        mg_model = model

    mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    mg_model.summary()
    if steps_per_epoch==None:
        steps_per_epoch = int(len(DataGenerator(**data_gen_args).return_data_list())/batch_size)
        val_steps_per_epoch = int(len(DataGenerator(**val_gen_args).return_data_list())/batch_size)
    else:
        val_steps_per_epoch = int(len(DataGenerator(**val_gen_args).return_data_list())/batch_size)


    # schedule = SGDRScheduler(min_lr=lr/1000,
    #                                  max_lr=lr,
    #                                  steps_per_epoch=steps_per_epoch,
    #                                  lr_decay=0.9,
    #                                  cycle_length=5,
    #                                  mult_factor=1.5)

    if val_data_dir=='':
        mg_model.fit_generator(training_generator,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)
    else:
        mg_model.fit_generator(training_generator,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               validation_steps=val_steps_per_epoch,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="data folder")

    parser.add_argument("--val_data_dir", type=str,
                        dest='val_data_dir',default='',
                        help="val data folder")


    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-5, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=0.1,
                        help="prior_lambda regularization parameter")

    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=None,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='', #default='../models/miccai2018_10_02_init1.h5'
                        help="optional h5 model file to initialize with")

    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="first epoch")


    args = parser.parse_args()
    train(**vars(args))

    # python train_vae.py /home/huang/cs236_project/data/preprocessed_connectomes/training_data/ --val_data_dir /home/huang/cs236_project/data/preprocessed_connectomes/val_data/ --model_dir /home/huang/cs236_project/models/vae/ --gpu 1 --prior_lambda 1 --batch_size 16
