"""VAE that creates goal distribution."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_util as ptu

class VAEGoal(nn.Module):
    def __init__(self,
                 log_dir,
                 input_size=256,
                 num_skills=1,
                 code_dim=100,
                 beta=0.5,
                 lr=1e-3,
                 ):
        """Initialize the density model.

        Args:
          num_skills: number of densities to simultaneously track
        """
        super().__init__()
        self._num_skills = num_skills
        
        self.latent_dim = 150

        input_dim = np.prod(input_size)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 150),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 150),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 150),
            nn.ReLU(),
            nn.Linear(150, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, input_dim),
        )

        self.lr = 0.001 
        self.beta = 0.5
        self.params = (list(self.enc.parameters()) +
                  list(self.enc_mu.parameters()) +
                  list(self.enc_logvar.parameters()) +
                  list(self.dec.parameters()))
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.input_size = input_size
        import os
        self.savedir = log_dir
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def get_output_for(self, aug_obs, sample=True):
        """
        Returns the log probability of the given observation.
        """
        obs = aug_obs
        with torch.no_grad():
            enc_features = self.enc(obs)
            mu = self.enc_mu(enc_features)
            logvar = self.enc_logvar(enc_features)

            stds = (0.5 * logvar).exp()
            if sample:
                epsilon = ptu.randn(*mu.size())
            else:
                epsilon = torch.ones_like(mu)
            code = epsilon * stds + mu

            obs_distribution_params = self.dec(code)
            log_prob = (obs - obs_distribution_params)**2
            log_prob = torch.sum(log_prob, -1, keepdim=True)
       # only returns the negative log reconstruction diff
        return log_prob

    def update(self, aug_obs):
        obs = aug_obs
        enc_features = self.enc(obs)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        code = epsilon * stds + mu

        kle = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()
              
        obs_distribution_params = self.dec(code)
        log_prob = -1. * F.mse_loss(obs[:, :, :], obs_distribution_params,
                                    reduction='mean')
        # update on the loss (kle + negative log reconstruct loss)
        loss = self.beta * kle - log_prob

        return loss
