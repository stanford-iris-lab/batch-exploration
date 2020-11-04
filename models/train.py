from models.img_vae import SimpleVAE
from models.transition_model import TransitionModel
from models.goal_vae import VAEGoal
from models.discriminator import BinClassifier

import pytorch_util as ptu
import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import torch.optim as optim

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Train latent dynamics model """
def train_dynamics_model(dynamics_model, obs_z, next_z, actions, dyn_optimizer):
    dynamics_model.train()
    dynamics_model.float()
    transition_loss, pred_z = dynamics_model.update(obs_z, next_z, actions)
    dyn_optimizer.zero_grad()
    transition_loss.backward(retain_graph=True)
    dyn_optimizer.step()
    return transition_loss.cpu().detach().item(), pred_z

""" Train VAE on goals and non goals """
def train_vae(args, enc_dec, obs, goals, enc_optimizer, beta):
    enc_losses = []
    ''' Train on non-goals '''
    vae_ipt = obs[:, 0,:,:,:].clone()
    vae_out = vae_ipt.clone()
    vae_ipt = pix_augment(vae_ipt)
    rec, _, _, klloss, obs_z = enc_dec.forward(vae_ipt.unsqueeze(1))
    rec_loss = ((rec.squeeze(1) - vae_out)**2).mean()
    vae_loss = beta * klloss + rec_loss
    enc_losses.append(vae_loss)
    
    ''' Train on goals '''
    g_rec = None
    if not args.vae_no_goal:
        goals_copy = goals.clone()
        g_out = goals_copy.clone()
        g_inpt = pix_augment(goals_copy)
        g_rec, _, _, klloss, goal_z = enc_dec.forward(g_inpt.unsqueeze(1)) 
        rec_loss = ((g_rec.squeeze(1) - g_out)**2).mean()
        goal_loss = beta * klloss + rec_loss
        enc_losses.append(goal_loss)
    
    enc_loss = torch.stack(enc_losses).sum()
    enc_optimizer.zero_grad()
    enc_loss.backward(retain_graph=True)
    enc_optimizer.step()
    return enc_loss.cpu().detach().item(), g_rec, rec.cpu().detach()

""" Train dynamics models for disagreement ensemble """
def train_disgrmt_ensemble(img_buffer, dynamics_models, enc_dec, dyn_optimizer, batch_sz, predlen):
    dyn_losses = []
    for model in dynamics_models:
        obs, next_obs, actions, success = img_buffer.draw_samples(batch_size=batch_sz, length=predlen)
        obs = torch.tensor(obs).float().to(device) 
        next_obs = torch.tensor(next_obs).float().to(device)
        actions = torch.tensor(actions).float().to(device)
        _, _, _, _, obs_z = enc_dec.forward(obs)
        _, _, _, _, next_z = enc_dec.forward(next_obs)
        model.train()
        model.float()
        loss, _ = model.update(obs_z[:,:predlen,:], next_z[:,:predlen,:], actions[:,:predlen,:])
        dyn_losses.append(loss)
    dyn_losses = torch.stack(dyn_losses).sum()
    dyn_optimizer.zero_grad()
    dyn_losses.backward()
    dyn_optimizer.step()
    # to be comparable with a single dynamics model loss
    return dyn_losses.cpu().detach().item()/len(dynamics_models)

""" Train SMM density models """
# THESE ARE NON CROPPED IMAGES --> through ENCODER
def train_smm_density_models(density_vae, goal_vae, obs_z, goal_z, d_optimizer, g_optimizer):
    density_loss = density_vae.update(obs_z) 
    d_optimizer.zero_grad()
    density_loss = density_loss.mean()
    density_loss.backward(retain_graph=True)
    d_optimizer.step()

    goal_loss = goal_vae.update(goal_z)
    g_optimizer.zero_grad()
    goal_loss = goal_loss.mean()
    goal_loss.backward(retain_graph=True)
    g_optimizer.step() 
    return (density_loss.cpu().detach().item() + goal_loss.cpu().detach().item()) / 2.

def train_classifiers(classifiers, enc_dec, obs, goals, c_optimizer, batch_sz, score_path=None):
    classifier_losses = []
    obs = obs.reshape(-1, 3, 64, 64)
    for classifier in classifiers:
        ng_ind = np.random.randint(0, len(obs), batch_sz)
        obs_samples = obs[ng_ind, :, :, :].clone()
        obs_samples = pix_augment(obs_samples)
        _, _, _, _, ng_z = enc_dec.forward(obs_samples.unsqueeze(1))
        ng_z = ng_z.squeeze(1)
        g_ind = np.random.randint(0, len(goals), batch_sz)
        g_samples = goals[g_ind, :, :, :].clone()
        g_samples = pix_augment(g_samples)
        _, _, _, _, g_z = enc_dec.forward(g_samples.unsqueeze(1))
        g_z = g_z.squeeze(1)
        xs = torch.cat((ng_z, g_z), 0)
        ys = torch.cat((torch.zeros(ng_z.size(0)), torch.ones(g_z.size(0))), 0).float().to(device)
        classifier.train()
        y_hat = classifier.forward(xs, calc_loss=True)
        loss = classifier.criterion(y_hat.squeeze(1), ys)
        if score_path is not None:
            # save classifier scores for goals and non goals here
            pred_scores = y_hat.clone()
            pred_scores = torch.sigmoid(pred_scores)
            pred_scores = ptu.get_numpy(pred_scores)
            pred_neg = pred_scores[:len(pred_scores)//2]
            pred_pos = pred_scores[len(pred_scores)//2:]
            np.savetxt(score_path + '/non_goals.txt', pred_neg)
            np.savetxt(score_path + '/goals.txt', pred_pos)
        if torch.isnan(loss):
            raise ValueError('Classifier loss nan')
        classifier_losses.append(loss)
        ''' Mixup '''
        shuffled_xs = xs.clone()
        shuffled_x = shuffled_xs[torch.randperm(shuffled_xs.size(0))]
        lams = torch.tensor(np.array([np.random.beta(1, 1) for _ in range(batch_sz)])).to(device).unsqueeze(1).float()
        x = torch.tensor(lams * shuffled_xs[:batch_sz] + (1. - lams) * shuffled_xs[batch_sz:])
        x_out = classifier.forward(x, calc_loss=True) # output of mixed inputs
        y_out = lams * classifier(shuffled_xs[:batch_sz]) + (1. - lams) * classifier(shuffled_xs[batch_sz:]) # mixed classifier outputs
        loss = classifier.criterion(x_out, y_out)
        if torch.isnan(loss):
            raise ValueError('Mixup classifier loss nan')
        classifier_losses.append(0.1 * loss) # Weight mixup by .1
    
    classifier_losses = torch.stack(classifier_losses).sum()
    c_optimizer.zero_grad()
    classifier_losses.backward(retain_graph=True) 
    c_optimizer.step()
    return classifier_losses.cpu().detach().item()


def train_img_classifiers(classifiers, obs, goals, c_optimizer, batch_sz, score_path=None):
    classifier_losses = []
    obs = obs.reshape(-1, 3, 64, 64)
    for classifier in classifiers:
        ng_ind = np.random.randint(0, len(obs), batch_sz)
        obs_samples = obs[ng_ind, :, :, :].clone()
        # obs_samples = pix_augment(obs_samples).squeeze(1)
        g_ind = np.random.randint(0, len(goals), batch_sz)
        g_samples = goals[g_ind, :, :, :].clone()
        # g_samples = pix_augment(g_samples).squeeze(1)
        xs = torch.cat((obs_samples, g_samples), 0)
        ys = torch.cat((torch.zeros(obs_samples.size(0)), torch.ones(g_samples.size(0))), 0).float().to(device)
        classifier.train()
        y_hat = classifier.forward(xs, calc_loss=True).squeeze(1)
        loss = classifier.criterion(y_hat.squeeze(1), ys)
        if score_path is not None:
            # save classifier scores for goals and non goals here
            pred_scores = y_hat.clone()
            pred_scores = torch.sigmoid(pred_scores)
            pred_scores = ptu.get_numpy(pred_scores)
            pred_neg = pred_scores[:len(pred_scores)//2]
            pred_pos = pred_scores[len(pred_scores)//2:]
            np.savetxt(score_path + '/non_goals.txt', pred_neg)
            np.savetxt(score_path + '/goals.txt', pred_pos)
        if torch.isnan(loss):
            raise ValueError('Classifier loss nan')
        classifier_losses.append(loss)
    classifier_losses = torch.stack(classifier_losses).mean()
    c_optimizer.zero_grad()
    classifier_losses.backward(retain_graph=True) 
    c_optimizer.step()
    return classifier_losses.cpu().detach().item()
    
