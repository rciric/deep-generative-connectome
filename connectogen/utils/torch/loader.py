# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generic trainer for torch networks.
"""


import os, re
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import squareform


root = '/Users/rastko/Downloads/preprocessed_connectomes/derivatives/'


class ConnectomeDataLoader(object):
    # TODO: add test/train split
    """
    Data loader for connectome data.

    Attributes
    ----------
    """
    def __init__(self,
                 root,
                 parc,
                 res,
                 batch_size=30,
                 shuffle=False,
                 vectorised=True):
        """Initialise a data loader for connectome data.

        Parameters
        ----------
        root: str
            Root directory in which connectome data are stored.
        parc: str
            Parcellation over which connectome data are defined.
        res: int
            Connectome resolution.
        batch_size: int
            Number of samples to draw at a time.
        shuffle: bool
            Indicates whether the data should be shuffled at the end of each
            epoch.
        vectorised: bool
            Indicates whether the data to be loaded is in vectorised form.
        """
        self.tasks = defaultdict(int)
        self.subs = defaultdict(int)
        self.n_tasks = 0
        self.n_subs = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vectorised = vectorised
        self.classify = 'task'

        files = [os.path.join(path, name)
                 for path, subdirs, files in os.walk(root)
                 for name in files]
        matches = [i for i in files if re.match(
                      '.*desc-{}{}.*connectome*'.format(parc, res), i
                  )]
        self.data = [self.import_data(path) for path in matches]
        self.data = [i for i in self.data if i]
        self.tasks = {v: k for k, v in self.tasks.items()}
        self.subs = {v: k for k, v in self.subs.items()}
        self.n_batches = len(self.data) // self.batch_size
        self.batch_size_final = len(self.data) % self.batch_size
        self.n = 0
        self.batch_data()

    def __len__(self):
        if self.batch_size_final > 0:
            return self.n_batches + 1
        else:
            return self.n_batches

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_batches:
            return self.sample(increment=True)
        elif self.batch_size_final > 0 and self.n == self.n_batches:
            return self.sample(final=True, increment=True)
        else:
            raise StopIteration

    def import_data(self, path):
        try:
            x = pd.read_csv(path, header=None, sep='\t').values.squeeze()
        except Exception:
            return None
        try:
            if np.any(np.isnan(x)):
                return None
        except TypeError:
            print(path)
        if not self.vectorised:
            x = squareform(x - np.eye(x.shape[0]))
        y = re.search('(?<=task-)[^_]+', path)
        y = y.string[y.start():y.end()]
        if not self.tasks[y]:
            self.n_tasks += 1
            self.tasks[y] = self.n_tasks
        y = self.tasks[y]
        z = re.search('(?<=sub-)[^_]+', path)
        z = z.string[z.start():z.end()]
        if not self.subs[z]:
            self.n_subs += 1
            self.subs[z] = self.n_subs
        z = self.subs[z]
        return x, y, z

    def batch_data(self):
        if self.shuffle:
            self.order = torch.randperm(len(self.data))
        else:
            self.order = torch.arange(len(self.data))

    def sample(self, increment=False, final=False):
        start = self.n * self.batch_size
        end = (self.n + 1) * self.batch_size
        if increment: self.n += 1
        samples = self.order[start:end]
        if self.classify == 'task':
            batch = (torch.Tensor([self.data[s][0] for s in samples]),
                     torch.Tensor([self.data[s][1] for s in samples]))
        elif self.classify == 'subject':
            batch = (torch.Tensor([self.data[s][0] for s in samples]),
                     torch.Tensor([self.data[s][2] for s in samples]))
        if final:
            self.batch_data()
        return batch
