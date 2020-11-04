from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.optim as optim

#### MODELS USED

## VAE
class SimpleVAE(nn.Module):
  def __init__(self, device, hidden_size, log_dir, activation_function='relu',ch=3):
    super().__init__()
    self.hidden_size = hidden_size
    self.enc = VisualEncoderAttn(hidden_size, ch=ch)
    self.dec = VisualReconModel(hidden_size, None)
    self.device = device
    self.sigmoid = nn.Sigmoid()
    self.params = (list(self.enc.parameters()) +
                  list(self.dec.parameters()))
    self.optimizer = optim.Adam(self.params, lr=0.001)
    self.savedir = log_dir + '/enc_dec'
    import os 
    if not os.path.exists(self.savedir):
        os.mkdir(self.savedir)
    
  def forward(self, observation, reconstruct=True):
    hidden = self.enc(observation)
    mu, std = hidden[:, :, :self.hidden_size], hidden[:, :, self.hidden_size:]
    std = std.clamp(min=1e-5)
    samples = torch.empty(mu.shape).normal_(mean=0,std=1).to(self.device) 
    encoding = mu + std * samples

    p = torch.distributions.Normal(mu, std)
    mu2 = torch.zeros((mu.shape)).to(self.device)
    std2 = torch.ones((std.shape)).to(self.device)
    q = torch.distributions.Normal(mu2, std2)

    klloss = torch.distributions.kl_divergence(p, q).mean()

    if reconstruct:
      rec = self.dec(encoding)
    else:
      rec = None
    return rec, mu, std, klloss, encoding


## Image Encoder
class VisualEncoderAttn(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, hidden_size, activation_function='relu', ch=3):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.softmax = nn.Softmax(dim=2)
    self.sigmoid = nn.Sigmoid()
    self.ch = ch
    self.conv1 = nn.Conv2d(self.ch, 32, 4, stride=2)
    self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 2* hidden_size)

  def forward(self, observation):
    batchsize, trajlen = observation.size(0), observation.size(1)
    self.width = observation.size(3)
    observation = observation.view(trajlen*batchsize, 3, self.width, 64) #6, 64
    if self.ch == 3:
      observation = observation[:, :3, :, :]
    
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv1_1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv2_1(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv3_1(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = self.act_fn(self.conv4_1(hidden))
    
    hidden = hidden.reshape(trajlen*batchsize, -1)
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.fc2(hidden)
    hidden = hidden.reshape(batchsize, trajlen, -1)
    return hidden
  
## Reconstruction Model
class VisualReconModel(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, hidden_size, action_size, activation_function='relu', action_len=5):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(hidden_size * 1, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)
    self.sigmoid = nn.Sigmoid()
    self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
    self.conv1_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv2_1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv3_1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
    self.conv4_1 = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)

  def forward(self, hidden):
    batchsize, trajlen = hidden.size(0), hidden.size(1)
    hidden = hidden.reshape(trajlen*batchsize, -1)
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    hidden = hidden.view(-1, 128, 1, 1)

    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    residual = self.sigmoid(self.conv4(hidden))
    
    residual = residual.view(batchsize, trajlen, residual.size(1), residual.size(2), residual.size(3))
    return residual
