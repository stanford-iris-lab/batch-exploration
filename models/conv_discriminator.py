from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
#### MODELS USED

## VAE
class ConvClassifier(nn.Module):
  def __init__(self, log_dir, hidden_size=256, activation_function='relu',ch=3):
    super().__init__()
    self.hidden_size = hidden_size
    self.enc = VisualEncoderAttn(hidden_size, ch=ch)
    self.lin = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                )
    self.sigmoid = nn.Sigmoid()
    self.params = (list(self.enc.parameters()) +
                  list(self.lin.parameters()))
    self.optimizer = optim.Adam(self.params, lr=0.001)
    self.criterion = nn.BCEWithLogitsLoss()
    self.savedir = log_dir  
    import os 
    if not os.path.exists(self.savedir):
        os.mkdir(self.savedir)
    
  def forward(self, observation, calc_loss=True):
    hidden = self.enc(observation)
    out = self.lin(hidden)
    if calc_loss:
        return out
    return self.sigmoid(out)

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
    self.fc2 = nn.Linear(512, hidden_size)

  def forward(self, observation):
    batchsize = observation.size(0)
    trajlen = 1
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
  
