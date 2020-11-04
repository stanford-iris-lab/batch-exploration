from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.optim as optim

## f_dec, model of dynamics in latent space
class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, hidden_size, action_size, log_dir, num=None, recurrent=True, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.recurrent = recurrent
        if recurrent:
            self.fc1 = nn.LSTM(hidden_size + action_size, 128, 1)
        else:
            self.fc1 = nn.Linear(hidden_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, hidden_size)
        self.params = (list(self.fc1.parameters()) + 
                       list(self.fc2.parameters()) + 
                       list(self.fc3.parameters()) + 
                       list(self.fc4.parameters()))
        self.optimizer = optim.Adam(self.params, lr=0.001)
        if num is not None:
            self.savedir = log_dir + '/dynamics_model{}'.format(num)
        else:
            self.savedir = log_dir + '/dynamics_model'
        import os
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def forward(self, ipt, action, hidden=None):
        ipt = torch.cat([ipt, action], dim=-1)
        batchsize, trajlen = ipt.size(0), ipt.size(1)
        ipt.view(-1, ipt.size(2))
        if self.recurrent:
            ipt, hidden = self.fc1(ipt)
        else:
            ipt = self.fc1(ipt)
        ipt = self.act_fn(ipt)
        ipt = self.act_fn(self.fc2(ipt))
        ipt = self.act_fn(self.fc3(ipt))
        ipt = self.fc4(ipt)
        out = ipt.view(batchsize, trajlen, -1)
        return out, hidden
    
    def predict(self, prev_hidden, actions):
        all_losses = []
        next_step = []
        # State latent encoding
        next_step_encoding = prev_hidden[:, 0:1, :]
        next_step.append(next_step_encoding)
        # Length of trajectory predicted
        predlen = 10
        if prev_hidden.shape[1] > 1 and prev_hidden.shape[1] < 10:
            predlen = prev_hidden.shape[1]
        
        hidden = None
        ## Rolling out forward latent dynamics
        for p in range(predlen):
            this_act = actions[:, p:p+1, :]
            next_step_encoding, hidden = self.forward(next_step_encoding, this_act, hidden)
            next_step.append(next_step_encoding)
            ## All predicted latent states
        next_step = torch.cat(next_step, 1)
        return next_step[:, 1:, :]
            
    def update(self, prev_hidden, next_hidden, actions):
        hidden = self.predict(prev_hidden, actions)
        loss_op = nn.MSELoss()
        loss = loss_op(hidden, next_hidden)
        return loss, hidden

