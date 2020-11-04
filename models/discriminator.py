"""Binary classifier that classifies whether a state is a goal state or not."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
import pytorch_util as ptu

class BinClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 log_dir,
                 ith,
                 lr=1e-3,
                 ):
        """Initialize each binary classifier.

        """
        super().__init__()

        input_dim = np.prod(input_size)
        self.l1 = spectral_norm(nn.Linear(input_dim, 128))
        self.l2 = spectral_norm(nn.Linear(128, 64))
        self.l3 = spectral_norm(nn.Linear(64, 64))
        self.l4 = spectral_norm(nn.Linear(64, 1))
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.lr = 0.001 
        self.params = list(self.l1.parameters()) + list(self.l2.parameters()) + list(self.l3.parameters()) + list(self.l4.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.input_size = input_size
        self.criterion = nn.BCEWithLogitsLoss()
        import os
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.root = log_dir
        self.savedir = log_dir + '/{}'.format(ith)
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def forward(self, aug_obs, calc_loss=False):
        x = self.relu(self.l1(aug_obs))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)
        if calc_loss:
            return x
            # return self.pred(aug_obs) # don't add sigmoid b/c LogitsLoss criterion already does
        return self.sig(x)
        # return self.sig(self.pred(aug_obs))
        
class BinClassifier_InsNorm(nn.Module):
    def __init__(self,
                 input_size,
                 log_dir,
                 ith,
                 lr=1e-3,
                 ):
        """Initialize each binary classifier.

        """
        super().__init__()

        input_dim = np.prod(input_size)
        # apply instance norm before relu activation, final sigmoid
        self.l1 = nn.Linear(input_dim, 128)
        self.norm1 = nn.InstanceNorm1d(128)
        self.l2 = nn.Linear(128, 64)
        self.norm2 = nn.InstanceNorm1d(64)
        self.l3 = nn.Linear(64, 64)
        self.norm3 = nn.InstanceNorm1d(64)
        self.l4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.lr = 0.001 
        self.params = list(self.l1.parameters()) + list(self.l2.parameters()) + list(self.l3.parameters()) + list(self.l4.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.input_size = input_size
        self.criterion = nn.BCEWithLogitsLoss()
        import os
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.root = log_dir
        self.savedir = log_dir + '/{}'.format(ith)
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def forward(self, aug_obs, calc_loss=False):
        x = self.l1(aug_obs)
        x = self.relu(self.norm1(x.unsqueeze(1)).squeeze(1))
        x = self.l2(x)
        x = self.relu(self.norm2(x.unsqueeze(1)).squeeze(1))
        x = self.l3(x)
        x = self.relu(self.norm3(x.unsqueeze(1)).squeeze(1))
        x = self.l4(x)
        if calc_loss:
            return x # don't add sigmoid b/c LogitsLoss criterion already does
        return self.sig(x)
        
