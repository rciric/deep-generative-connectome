# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multilayer perceptrons
~~~~~~~~~~~~~~~~~~~~~~
Interfaces for building simple fully connected neural networks.
"""
import torch
from torch import nn
from connectogen.utils.torch.utils import _listify


class MLPNetwork(nn.Module):
    """
    Generalised deep fully connected architecture. Inexplicably seems to run
    faster than a full-field-filter convolution.

    Attributes
    ----------
    n_hidden: int
        Number of hidden layers.
    """
    def __init__(self,
                 in_dim=4095,
                 out_dim=128,
                 hidden=(256, 256),
                 nonlinearity='leaky',
                 batch_norm=True,
                 dropout=0,
                 bias=True,
                 leak=0.2):
        """Initialise a deep fully connected network.
        
        Parameters
        ----------
        hidden: tuple
            Tuple denoting the number of units in each hidden layer.
        nonlinearity: tuple
            Nonlinearity to use in each hidden layer.
        batch_norm: bool or tuple
            Indicates whether batch normalisation should be applied to each
            layer.
        dropout: float or tuple
            Dropout probability of units in each hidden layer.
        bias: bool or tuple
            Indicates whether each convolutional filter includes bias terms
            for each unit.
        leak: float
            Slope of the negative part of the hidden layers' leaky ReLU
            activation function. Used only if a leaky ReLU activation function
            is specified.
        """
        super(MLPNetwork, self).__init__()
        hidden = _listify(hidden)
        layers = [in_dim] + hidden + [out_dim]
        self.n_hidden = len(hidden) + 1
        self.hidden = nn.ModuleList()

        nonlinearity = _listify(nonlinearity, self.n_hidden)
        batch_norm = _listify(batch_norm, self.n_hidden)
        dropout = _listify(dropout, self.n_hidden)
        bias = _listify(bias, self.n_hidden)

        for i, (r, s) in enumerate(zip(layers[1:], layers[:-1])):
            layer = [
                nn.Linear(in_features=s,
                          out_features=r,
                          bias=bias[i])
            ]

            if batch_norm[i]:
                layer.append(nn.BatchNorm1d(r))

            if nonlinearity[i] == 'leaky':
                layer.append(nn.LeakyReLU(negative_slope=leak, inplace=True))
            elif nonlinearity[i] == 'relu':
                layer.append(nn.ReLU(inplace=True))
            elif nonlinearity[i] == 'sigmoid':
                layer.append(nn.Sigmoid())
            elif nonlinearity[i] == 'tanh':
                layer.append(nn.Tanh())

            if dropout[i] != 0:
                layer.append(nn.Dropout(dropout[i]))

            self.hidden.append(nn.Sequential(*layer))

    def forward(self, x):
        for i in range(self.n_hidden):
            x = self.hidden[i](x)
        return x
