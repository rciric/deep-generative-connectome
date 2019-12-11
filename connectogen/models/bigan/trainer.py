# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
BiGAN trainer
~~~~~~~~~~~~~
Trainer class for the BiGAN
"""
import torch
from torch import nn, optim
from connectogen.utils.torch.utils import thumb_grid, animate_gif
from connectogen.utils.torch.trainer import ConnectomeTrainer


def config_targets(batch_size, cuda=False):
    """
    Configure the targets for the BiGAN discriminator.

    Parameters
    ----------
    batch_size: int
        Number of observations per batch.
    cuda: bool
        Indicates whether the targets should be placed on CUDA.

    Returns
    -------
    generator_target: Tensor
        Target indicating that the disciminator predicts the input (latent,
        manifest) pair was processed by the generator.
        This is the discriminator's target for generator-processed pairs and
        the generator-encoder's target for encoder-processed pairs.
    encoder_target: Tensor
        Target indicating that the disciminator predicts the input (latent,
        manifest) pair was processed by the encoder.
        This is the discriminator's target for encoder-processed pairs and
        the generator-encoder's target for generator-processed pairs.
    """
    generator_target = torch.ones(batch_size, 1)
    encoder_target = torch.zeros(batch_size, 1)
    if cuda:
        generator_target = generator_target.cuda()
        encoder_target = encoder_target.cuda()
    return generator_target, encoder_target


class BiGANTrainer(ConnectomeTrainer):
    """
    Trainer class for an adversarially learned inference network (BiGAN).

    Attributes
    ----------
    loader: DataLoader
        DataLoader for the dataset to be used for training.
    model: Module
        BiGAN to be trained.
    batch_size: int
        Number of observations per mini-batch.
    learning_rate: float
        Optimiser learning rate.
    max_epoch: int
        Number of epochs of training.
    cuda: bool
        Indicates whether the model should be trained on CUDA.
    """
    def __init__(self,
                 loader,
                 model,
                 batch_size=30,
                 learning_rate=0.0002,
                 max_epoch=20,
                 cuda=False):
        super(BiGANTrainer, self).__init__(loader, model, batch_size,
                                           learning_rate, max_epoch, cuda)

        self.optimiser_d = optim.Adam(self.model.discriminator.parameters(),
                                      lr=self.learning_rate)
        self.optimiser_g = optim.Adam(self.model.generator.parameters(),
                                      lr=self.learning_rate)
        self.optimiser_e = optim.Adam(self.model.encoder.parameters(),
                                      lr=self.learning_rate)

        self.loss = nn.BCEWithLogitsLoss()

    def train(self, log_progress=True, save_images=True,
              log_interval=100, img_prefix='bigan', gd_rep=2):
        """Train a BiGAN.

        Parameters
        ----------
        . . .
        """
        self.model.train()
        z_probe = self.model.prior.sample([16, self.model.latent_dim])
        if self.cuda: z_probe = z_probe.cuda()
        if save_images:
            save = -1
            image_inst = '{}'.format(img_prefix) + '_{epoch:03d}.png'
            image_inst_f = '{}'.format(img_prefix) + '_{epoch}.png'
            image_out = '{}.gif'.format(img_prefix)
            self._save_images(z_probe, save, image_inst)

        for epoch in range(self.max_epoch):
            loss_d_epoch = 0
            loss_g_epoch = 0
            loss_e_epoch = 0
            for i, (x, _) in enumerate(self.loader):
                batch_size = x.size(0)
                (self.target_g,
                 self.target_e) = config_targets(batch_size,
                                                 cuda=self.cuda)
                z = self.model.prior.sample(
                    [batch_size, self.model.latent_dim])
                if self.cuda:
                    x = x.cuda()
                    z = z.cuda()

                x_hat = self.model.generator(z).detach()
                z_hat = self.model.encoder(x).detach()
                error_d, _, _ = self.train_discriminator(
                    generator_data=(z, x_hat),
                    encoder_data=(z_hat, x)
                )

                for _ in range(gd_rep):
                    x_hat = self.model.generator(z)
                    z_hat = self.model.encoder(x)
                    error_g, error_e = self.train_generator_encoder(
                        generator_data=(z, x_hat),
                        encoder_data=(z_hat, x)
                    )
                    loss_g_epoch += error_g / gd_rep
                    loss_e_epoch += error_e / gd_rep
                loss_d_epoch += error_d

            if save_images:
                save += 1
                self._save_images(z_probe, save, image_inst)
            if log_progress:
                self.report(epoch, loss_d_epoch / i, 'Discriminator')
                self.report(epoch, loss_g_epoch / i, 'Generator')
                self.report(epoch, loss_e_epoch / i, 'Encoder')
        if save_images:
            animate_gif('animated_bigan_training.gif', src_fmt=image_inst_f)

    def train_discriminator(self, generator_data, encoder_data):
        """Evaluate the error of the InfoBiGAN's discriminator network for a
        single mini-batch of generator- and encoder-processed data.

        Parameters
        ----------
        generator_data: Tensor
            Mini-batch of observations sampled from the generator.
        encoder_data: Tensor
            Mini-batch of observations sampled from the encoder.
        """
        self.optimiser_d.zero_grad()

        prediction_g = self.model.discriminator(*generator_data)
        error_g = self.loss(prediction_g, self.target_g)
        error_g.backward()

        prediction_e = self.model.discriminator(*encoder_data)
        error_e = self.loss(prediction_e, self.target_e)
        error_e.backward()

        self.optimiser_d.step()

        return error_g + error_e, prediction_g, prediction_e

    def train_generator_encoder(self, generator_data, encoder_data):
        """Evaluate the error of the BiGAN's generator and encoder
        networks for mini-batch of data sourced from the generator
        z -> x_hat and the encoder x -> z_hat.

        Parameters
        ----------
        generator_data: Tensor
            Mini-batch of observations sampled from the generator.
        encoder_data: Tensor
            Mini-batch of observations sampled from the encoder.
        """
        self.optimiser_g.zero_grad()
        prediction_g = self.model.discriminator(*generator_data)
        g_loss = self.loss(prediction_g, self.target_e)
        g_loss.backward()
        self.optimiser_g.step()

        self.optimiser_e.zero_grad()
        prediction_e = self.model.discriminator(*encoder_data)
        e_loss = self.loss(prediction_e, self.target_g)
        e_loss.backward()
        self.optimiser_e.step()

        return g_loss, e_loss

    def report(self, epoch, loss, name=''):
        """Print a report on the current progress of training.

        Parameters
        ----------
        epoch: int
            Current epoch.
        loss: Tensor
            Output of the loss function.
        name: str
            Name of the loss function (if there is more than one for the
            current network).
        """
        print('Epoch [{}/{} ({:.0f}%)]\t'
              '{} Loss [{:.6f}]'.format(
                  epoch + 1, self.max_epoch,
                  100 * (epoch + 1) / self.max_epoch,
                  name, loss.item()))

    def _save_images(self, z, save, image_fmt):
        """Save thumbnails of images generated during training."""
        probe_gen = self.squareform(self.model.generator(z))
        n, h, w = probe_gen.size()
        probe_gen = probe_gen.view(n, 1, h, w)
        thumb_grid(probe_gen, save=True, cuda=self.cuda, cmap='coolwarm',
                   vals=(-0.6, 0.6), file=image_fmt.format(epoch=save + 1))
