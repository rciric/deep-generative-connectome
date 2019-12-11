# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
BiGAN
~~~~~
Adversarially learned inference network
"""
import torch
from torch import nn
from connectogen.modules.torch.mlp import MLPNetwork
from connectogen.utils.torch.utils import eps, _listify


class BiGAN(object):
    """An adversarially learned inference network (BiGAN).

    Attributes
    ----------
    discriminator: JointDiscriminator
        The BiGAN's discriminator network, which is presented a set of
        latent space-manifest space pairs and determines whether each
        pair was produced by the encoder or the generator.
    generator: MLPNetwork
        The BiGAN's generator network, which learns the underlying
        distribution of a dataset through a minimax game played against the
        discriminator.
    encoder: MLPNetwork
        The BiGAN's inferential network, which learns the latent space
        encodings of a dataset through a minimax game played against the
        discriminator.
    latent_dim: int
        Dimensionality of the latent space.
    """
    def __init__(self,
                 prior,
                 hidden=(256, 256),
                 bias=False,
                 manifest_dim=4950,
                 latent_dim=128):
        """Initialise an adversarially learned inference network (BiGAN).

        Parameters are ordered according to the discriminator and encoder
        networks. For instance, the second hidden parameter denotes the 
        number of units in the second hidden layer of D and E. The generator
        network currently uses an inverse architecture, so that the same
        parameter denotes the number of units in its second-to-last hidden
        layer.

        Parameters
        ----------
        prior: Distribution
            Distribution object with a `sample` method that takes as input
            matrix dimensions and produces as output samples from a
            distribution with the specified dimensions. Most torch
            Distributions can be used as priors.
        hidden: tuple
            Tuple denoting number of units in each hidden layer (other than
            the final/output layer).
        bias: bool or tuple
            Indicates whether each hidden layer includes bias terms for each
            unit.
        latent_dim: int
            Number of latent features that the generator network samples.
        manifest_dim: int
            Dimensionality of the observed (manifest) data.
        """
        n_hidden = len(hidden)

        self.prior = prior
        self.latent_dim = latent_dim
        self.manifest_dim = manifest_dim
        self.discriminator = JointDiscriminator(
            hidden=hidden, bias=bias, manifest_dim=self.manifest_dim,
            latent_dim=self.latent_dim)
        self.encoder = MLPNetwork(
            hidden=hidden, bias=bias, in_dim=manifest_dim,
            out_dim=latent_dim, batch_norm=False, dropout=[0] * (n_hidden - 1) + [0.5] * 2)
        self.generator = MLPNetwork(
            hidden=hidden, bias=bias, in_dim=latent_dim,
            out_dim=manifest_dim, batch_norm=False, dropout=[0.5] * 2 + [0] * (n_hidden - 1),
            nonlinearity=['leaky'] * (n_hidden + 1)) #+ ['tanh'])

    def train(self):
        self.discriminator.train()
        self.generator.train()
        self.encoder.train()

    def eval(self):
        self.discriminator.eval()
        self.generator.eval()
        self.encoder.eval()

    def cuda(self):
        self.discriminator.cuda()
        self.generator.cuda()
        self.encoder.cuda()

    def zero_grad(self):
        self.discriminator.zero_grad()
        self.generator.zero_grad()
        self.encoder.zero_grad()

    def load_state_dict(self, params_g, params_e, params_d):
        self.encoder.load_state_dict(params_e)
        self.generator.load_state_dict(params_g)
        self.discriminator.load_state_dict(params_d)


class JointDiscriminator(nn.Module):
    """A discriminator network that learns to identify whether a (latent,
    manifest) pair is drawn from the encoder or from the decoder.

    Attributes
    ----------
    x_discriminator: MLPNetwork
        Representational network for manifest-space data.
    z_discriminator: MLPNetwork
        Representational network for latent-space data.
    zx_discriminator: MLPNetwork
        Discriminator that splices together representations of latent- and
        manifest-space data and yields a decision regarding the provenance
        of the data pair.
    """
    def __init__(self,
                 manifest_dim=4950,
                 latent_dim=128,
                 hidden=(256, 256,),
                 bias=False):
        """Initialise a joint discriminator.

        Parameters
        ----------
        manifest_dim: int
            Side length of the input image.
        latent_dim: int
            Dimensionality of the latent space.
        hidden: tuple
            Tuple denoting the number of units in each hidden layer of
            the manifest-space representational network.
        bias: bool or tuple
            Indicates whether each hidden layer in the manifest representational
            network includes bias terms.
        """
        super(JointDiscriminator, self).__init__()
        self.x_discriminator = MLPNetwork(
            hidden=hidden, bias=bias, in_dim=manifest_dim,
            out_dim=latent_dim*2, batch_norm=False, dropout=0.5)
        self.z_discriminator = MLPNetwork(
            hidden=(latent_dim*2, latent_dim*2), bias=True,
            in_dim=latent_dim, out_dim=latent_dim*2, batch_norm=False, dropout=0.5)
        self.zx_discriminator = MLPNetwork(
            hidden=(latent_dim*4,), bias=True, in_dim=latent_dim*4,
            out_dim=1, batch_norm=False)

    def forward(self, z, x):
        z = self.z_discriminator(z)
        x = self.x_discriminator(x)
        zx = torch.cat([z, x], 1) + eps
        zx = self.zx_discriminator(zx)
        return zx

