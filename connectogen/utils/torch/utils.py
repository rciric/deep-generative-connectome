# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
General utilities.
"""
import os, glob
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.distributions import Categorical


eps = 1e-15


class UniformCategorical(Categorical):

    def __init__(self, n):
        super(UniformCategorical, self).__init__(
            probs=torch.Tensor([1.0 / n] * n))


def _listify(item, length=False):
    if length:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item] * length
        else:
            return list(item)
    else:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item]
        else:
            return list(item)


def adj2vec(adj, diag=False):
    """Reshape a symmetric adjacency matrix into a feature vector of edges in
    a differentiable manner. The resultant feature vector follows row-major
    order of the upper triangularised adjacency matrix.

    Based on work by Saurabh_Verma on the pytorch forums.

    Parameters
    ----------
    adj: Tensor
        Adjacency matrix from which edge weights are to be selected.
    diag: bool
        If True, indicates that the features on the diagonal should be
        appended to the end of the edge vector.
        If False (default), only off-diagonal features are included.

    Returns
    -------
    Tensor
        1-dimensional vector of edge weights in the symmetric adjacency matrix
        `adj`.
    """
    row_idx, col_idx = np.triu_indices(n=adj.shape[-1], k=1)
    vec = adj[torch.LongTensor(row_idx), torch.LongTensor(col_idx)]
    if diag:
        vec = torch.cat([vec, torch.diag(adj)])
    return vec


def vec2adj(vec, diag=False):
    """Reshape a feature vector of edges into a symmetric adjacency matrix in
    a differentiable manner. It is assumed that the feature vector corresponds
    to the upper triangle and is in row-major order.

    Parameters
    ----------
    vec: Tensor
        Edge weight vector to be folded into an adjacency matrix.
    diag: bool
        If True, indicates that the last n features of the edge vector are the
        diagonal elements of the adjacency matrix.
        If False, the edge vector is interpreted to include only off-diagonal
        features.
    """
    def _adj(vec, n):
        adj = torch.zeros((n, n))
        if vec.requires_grad:
            adj.requires_grad_()
            adj = adj.clone()
        row_idx, col_idx = np.triu_indices(n=n, k=1)
        adj[torch.LongTensor(row_idx), torch.LongTensor(col_idx)] = vec
        adj.t_()
        adj[torch.LongTensor(row_idx), torch.LongTensor(col_idx)] = vec
        adj.t_()
        return adj
    if diag:
        n = int(np.floor(np.sqrt(2 * len(vec))))
        row_idx, col_idx = np.diag_indices(n=n)
        adj = _adj(vec[:-n], n)
        adj[torch.LongTensor(row_idx), torch.LongTensor(col_idx)] = vec[-n:]
    else:
        n = int(np.ceil(np.sqrt(2 * len(vec))))
        adj = _adj(vec, n)
    return adj


def thumb_grid(im_batch, grid_dim=(4, 4), im_dim=(6, 6), cmap='bone',
               vals=None, save=False, file='example.png', cuda=False):
    """Generate a grid of image thumbnails.

    Parameters
    ----------
    im_batch: Tensor
        Tensor of dimensionality (number of images) x (channels)
        x (height) x (width)
    grid_dim: tuple
        Dimensionality of the grid where the thumbnails should be plotted.
    im_dim: tuple
        Size of the image canvas.
    save: bool
        Indicates whether the thumbnails should be saved as a single image.
    file: str
        File where the image should be saved if `save` is true.
    """
    fig = plt.figure(1, im_dim)
    grid = ImageGrid(fig, 111, nrows_ncols=grid_dim, axes_pad=0.05)
    if vals is not None:
        vmin, vmax = vals
    else:
        vmin = None; vmax = None
    for i in range(im_batch.size(0)):
        if cuda:
            img = im_batch[i, :, :, :].detach().cpu().numpy().squeeze()
        else:
            img = im_batch[i, :, :, :].detach().numpy().squeeze()
        grid[i].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(file, bbox_inches='tight')


def animate_gif(out, src_fmt, duration=0.1, delete=False):
    """Animate a GIF of the training process.

    Parameters
    ----------
    out: str
        Path where the animated image should be saved.
    src_fmt: str
        Generic path to source images to be used in the animation. Any
        instances of the string `{epoch}` will be replaced by the wildcard
        (`*`) and results sorted.
    duration: float
        Duration of each frame, in seconds.
    delete: bool
        Indicates whether the source images used to compile the GIF animation
        should be deleted.
    """
    print('[Animating]')
    files = sorted(glob.glob(src_fmt.format(epoch='*'))) 
    with imageio.get_writer(out, mode='I', duration=duration) as writer:
        for file in files:
            img = imageio.imread(file)
            writer.append_data(img)
            if delete:
                os.remove(file)
    print('[Animation ready]')
