import numpy as np
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy import hstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from matplotlib import pyplot
from tensorflow.python.framework import ops

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# define the standalone discriminator model
def define_discriminator(n_cat, in_shape=(4950,)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	d = Dense(4950, kernel_initializer=init,activation='relu')(in_image)
	d = Dense(2000, kernel_initializer=init,activation='relu')(d)
	d = Dense(1000, kernel_initializer=init,activation='relu')(d)
	d = Dense(450, kernel_initializer=init,activation='relu')(d)
	# real/fake output
	out_classifier = Dense(1, activation='sigmoid')(d)
	# define d model
	d_model = Model(in_image, out_classifier)
	# compile d model
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# create q model layers
	q = Dense(128)(d)
	#q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
	# q model output
	out_codes = Dense(n_cat, activation='softmax')(q)
	# define q model
	q_model = Model(in_image, out_codes)
	return d_model, q_model

# define the standalone generator model
def define_generator(gen_input_size):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image generator input
	in_lat = Input(shape=(gen_input_size,))
	# foundation for 7x7 image
	gen = Dense(100, kernel_initializer=init,activation='relu')(in_lat)
	gen = Dense(800, kernel_initializer=init,activation='relu')(gen)
	gen = Dense(1800, kernel_initializer=init,activation='relu')(gen)
	gen = Dense(2700, kernel_initializer=init,activation='relu')(gen)
	gen = Dense(4950, kernel_initializer=init,activation='relu')(gen)
	# tanh output
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model(in_lat, out_layer)
	return model
 
# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model, q_model):
	# make weights in the discriminator (some shared with the q model) as not trainable
	#d_model.trainable = False
	# connect g outputs to d inputs
	d_output = d_model(g_model.output)
	# connect g outputs to q inputs
	q_output = q_model(g_model.output)
	# define composite model
	model = Model(g_model.input, [d_output, q_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
	return model

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images and labels
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# select real samples
def select_real_samples(dataset, n_samples,i):
	# choose random instances
	#ix = randint(0, dataset.shape[0], n_samples)
	# select images and labels
	#X = dataset[ix]
	ind1 = i*n_samples
	ind2 = np.min([(i+1)*n_samples,len(dataset)])
	X = dataset[ind1:ind2,:]
	# generate class labels
	y = ones((len(X), 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_cat, n_samples):
	# generate points in the latent space
	z_latent = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_latent = z_latent.reshape(n_samples, latent_dim)
	# generate categorical codes
	cat_codes = randint(0, n_cat, n_samples)
	# one hot encode
	cat_codes = to_categorical(cat_codes, num_classes=n_cat)
	# concatenate latent points and control codes
	z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_cat, n_samples):
	# generate points in latent space and control codes
	z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, gan_model, latent_dim, n_cat, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_cat, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	#for i in range(100):
	#	# define subplot
	#	pyplot.subplot(10, 10, 1 + i)
	#	# turn off axis
	#	pyplot.axis('off')
	#	# plot raw pixel data
#		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	# filename1 = 'generated_plot_%04d.png' % (step+1)
	# pyplot.savefig(filename1)
	# pyplot.close()
	# save the generator model
	filename2 = 'gen_model_%04d.h5' % (step+1)
	g_model.save(filename2)
	# save the gan model
	filename3 = 'gan_model_%04d.h5' % (step+1)
	gan_model.save(filename3)
	#print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_cat, n_epochs, n_batch):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	dataset = dataset[np.random.permutation(len(dataset)),:]
	losses = np.zeros(shape=(n_steps,4))
	# manually enumerate epochs
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, n_batch)

#		X_real, y_real = select_real_samples(dataset, n_batch,i)
		# update discriminator and q model weights
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, n_batch)
		# update discriminator model weights
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the g via the d and q error
		_,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
		# summarize loss on this batch
		print('>%d, d[%.3f,%.3f], g[%.3f] q[%.3f]' % (i+1, d_loss1, d_loss2, g_1, g_2))
		losses[:,0] = d_loss1
		losses[:,1] = d_loss2
		losses[:,2] = g_1
		losses[:,3] = g_2
		# evaluate the model performance every 'epoch'
		if (i+1) % (bat_per_epo*5) == 0:
			print('epoch: ' + str(i+1))
			summarize_performance(i, g_model, gan_model, latent_dim, n_cat)
	return losses

# number of values for the categorical control code
n_cat = 10
# size of the latent space
latent_dim = 100
# create the discriminator
d_model, q_model = define_discriminator(n_cat)
# create the generator
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
# create the gan
gan_model = define_gan(g_model, d_model, q_model)
# load connectome
import scipy.io as scio
mat = scio.loadmat(file_name='msc_connectome_features.mat')
connectome_features = mat['connectome_features']
n_epochs = 500
n_batch = 64
mat2={}
losses = train(g_model, d_model, gan_model, connectome_features, latent_dim, n_cat, n_epochs, n_batch)
mat2['losses'] = losses
scio.savemat('losses.mat',mat2)
