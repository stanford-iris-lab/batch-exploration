from models.img_vae import SimpleVAE
from models.transition_model import TransitionModel
from models.goal_vae import VAEGoal
from models.discriminator import BinClassifier, BinClassifier_InsNorm
from models.conv_discriminator import ConvClassifier
from envs.manip_env.tabletop import Tabletop

from utils.utils import pix_augment, get_goal_imgs, get_obs 
from utils.logging import log_rankings 

from losses import Hist
from cem.cem import get_random_action_sequence, get_action_and_info, plan_actions 
from models.train import train_dynamics_model, train_vae, train_disgrmt_ensemble, train_smm_density_models, train_classifiers 

from replay_buffer.high_dim_replay import ImageBuffer
from replay_buffer.replay_buffer import ReplayBuffer

from configs.args import parse_args, create_log_dir

import pytorch_util as ptu
import pickle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time

from PIL import Image
import imageio
import cv2
import gtimer as gt
import copy
import json


## CONSTANTS ##
TOP_K = 5 # uniformly choose from the top K trajectories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    '''Initialize replay buffer, models, and environment.'''
    if args.logging > 0:
        import time
    replay_state_dim = 12
    if args.door == 1 or args.door == 3:
        replay_state_dim += 1
    elif args.door == 5:
        replay_state_dim += 1 + 3 * 2
    elif args.drawer:
        replay_state_dim += 3 * 3 + 1
    if not args.robot:
        replay_buffer = ReplayBuffer(
                                max_replay_buffer_size = args.replay_buffer_size,
                                trajectory_length=args.traj_length,
                                state_dim=replay_state_dim,
                                action_dim=args.action_dim,
                                savedir=args.log_dir,
                                )

    img_buffer = ImageBuffer( # 3x64x64 pixels
                                trajectory_length=args.num_traj_per_epoch*args.traj_length,
                                action_dim=args.action_dim,
                                savedir=args.log_dir,
                                memory_size=500,
                                )

    if args.logging == 0: # no env logging
        args.verbose = False
    if args.robot:
        import gym
        import franka_env
        env = gym.make("Franka-v0")
    else:
        env = Tabletop(
                    log_freq=args.env_log_freq, 
                    filepath=args.log_dir + '/env',
                    door=args.door,
                    drawer=args.drawer,
                    hard=args.hard,
                    verbose=args.verbose)

    if args.logging == 2:
        viz_env = Tabletop(
                    door=args.door,
                    drawer=args.drawer,
                    hard=args.hard,
                    log_freq=args.env_log_freq, 
                    filepath=None,
                    verbose=False)
    else:
        viz_env = None

    ''' Initialize models '''
    enc_dec = SimpleVAE(device, args.latent_dim, args.log_dir)
    if args.reload is not None:
        enc_dec.load_state_dict(torch.load(args.reload + '/enc_dec/{}model.bin'.format(args.reload_epoch)))
    enc_dec.to(device)
    enc_params = list(enc_dec.params)
    enc_optimizer = optim.Adam(enc_params, lr=1e-3) # just for enc_dec
     
    dynamics_models = None
    if args.dynamics_var:
        dynamics_models = []
        dyn_params = None
        for a in range(5):
            dynamics_model = TransitionModel(args.latent_dim, args.action_dim, args.log_dir, num=a)
            dynamics_model.to(device)
            dynamics_models.append(dynamics_model)
            if a == 0:
                dyn_params = list(dynamics_model.params)
            else:
                dyn_params += list(dynamics_model.params)
    else:
        dynamics_model = TransitionModel(args.latent_dim, args.action_dim, args.log_dir, recurrent=False)
        if args.reload is not None:
            dynamics_model.load_state_dict(torch.load(args.reload + '/dynamics_model/{}model.bin'.format(args.reload_epoch))) 
        dynamics_model.to(device)
        dyn_params = list(dynamics_model.params)
    dyn_optimizer = optim.Adam(dyn_params, lr=1e-3) # just for transition model

    # If using Classifiers
    classifiers = None
    if args.use_classifiers is not None:
        classifiers = []
        for i in range(args.num_classifiers):
            # classifier = BinClassifier(args.latent_dim, args.log_dir + '/classifier', i)
            if args.instance_normalized:
                classifier = BinClassifier_InsNorm(args.latent_dim, args.log_dir + '/classifier', i)
            else:
                classifier = BinClassifier(args.latent_dim, args.log_dir + '/classifier', i)
            if args.reload is not None:
                classifier.load_state_dict(torch.load(args.reload + '/classifier/{}/{}model.bin'.format(i, args.reload_epoch)))
            classifier.to(device)
            classifiers.append(classifier)
            if i == 0:
                c_params = list(classifier.params)
            else:
                c_params += list(classifier.params)
        c_optimizer = optim.Adam(c_params, lr=1e-3)
   
    # If using SMM
    density_vae = None
    goal_vae = None
    if args.smm:
        density_vae = VAEGoal(args.log_dir + '/density_model')
        goal_vae = VAEGoal(args.log_dir + '/goal')
        density_vae.to(device)
        goal_vae.to(device)
        d_params = list(density_vae.params)
        g_params = list(goal_vae.params)
        g_optimizer = optim.Adam(d_params, lr=1e-3) 
        d_optimizer = optim.Adam(g_params, lr=1e-3) 

    ''' Return goals '''
    goals = np.array(get_goal_imgs(args, env, filepath=args.log_dir + '/goal_ims')) 
    goals = goals / 255. 
    goals = torch.tensor(goals).float().to(device)
    goals = goals.permute(0, 3, 1, 2)

    # If flag 0, no training losses log
    if args.logging > 0:
        hist = Hist(args.log_dir)

    env.max_path_length = args.traj_length * args.num_traj_per_epoch
    
    # Clean the env memory to make sure above code isn't affecting the env
    if not args.robot:
        env.initialize()
    ob, env_info = None, None
    for epoch in gt.timed_for(range(args.num_epochs), save_itrs=True):
        if args.logging > 0:
            start = time.time()
        init_low_dim = None
        obs_sample = []
        high_dim_sample = []
        ob, env_info = env.reset_model(add_noise=args.add_noise)
        if epoch == 0 and args.logging > 0 and not args.robot: 
            init_array = env.get_obs() * 255.
            init_img = Image.fromarray(init_array.astype(np.uint8))
            init_img.save(args.log_dir + '/init.png')
        '''
        Log low dim state for plotting block interaction bars
        '''
        if not args.robot:
            init_low_dim = get_obs(args, env_info) 
            obs_sample.append(init_low_dim)

        init_ob = ob

        eps_obs = []
        eps_next = []
        eps_act = []

        for i in range(args.num_traj_per_epoch):
            ob = torch.tensor(ob).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            if i == 0: 
                high_dim_sample.append(ptu.get_numpy(ob.squeeze(0)))
                
            if epoch < 100:
                actions = get_random_action_sequence(env, args.traj_length, sample_sz = 1)
                actions = ptu.get_numpy(actions).squeeze(0)
            else:
                sorted_rewards, sorted_actions, sorted_preds = plan_actions(args, env, ob, dynamics_model, enc_dec, classifiers=classifiers, goal_vae=goal_vae, density_vae=density_vae, dynamics_models=dynamics_models)
                # Randomly select from the top K with highest reward
                act = np.random.choice(TOP_K)
                actions = sorted_actions[act]

                '''
                Log best and worst 3 trajectories (gifs and final state imgs)
                '''
                if args.logging > 0 and epoch % args.model_log_freq == 0:
                    log_rankings(args, enc_dec, hist.rankings_dir, viz_env, init_low_dim, sorted_actions, sorted_preds, epoch, i, sorted_rewards)

            action_sample = []
            for action in actions:
                # With low probability take a random action
                rand = np.random.uniform(0.0, 1.0)
                if rand < args.random_act_prob:
                    action = get_random_action_sequence(env, 1, sample_sz = 1).cpu().detach().numpy()
                    action = action.reshape(args.action_dim)

                next_ob, reward, terminal, env_info = env.step(action)
                ob = next_ob
                next_ob = torch.tensor(next_ob).permute(2, 0, 1).float().to(device).unsqueeze(0) # change to 3 x 64 x 64 obs
                high_dim_sample.append(ptu.get_numpy(next_ob.squeeze(0)))
                if not args.robot:
                    obs = get_obs(args, env_info)
                    obs_sample.append(obs) 
                    init_low_dim = obs_sample[-1].copy()
                action_sample.append(action)

            if not args.robot:
                replay_buffer.add_sample(
                        states=obs_sample[:-1],
                        next_states=obs_sample[1:],
                        actions=action_sample,
                        )
                last_obs = obs_sample[-1]

            eps_obs.append(high_dim_sample[:-1])
            eps_next.append(high_dim_sample[1:])
            eps_act.append(action_sample)
            
            last_frame = high_dim_sample[-1]
            obs_sample = []
            high_dim_sample = []
            # This becomes the init frame of the next traj
            if not args.robot:
                obs_sample.append(last_obs)
            high_dim_sample.append(last_frame)

        # reshape to -1, EPS SZ 50, 3, 64, 64
        eps_obs = np.array(eps_obs).reshape(-1, args.num_traj_per_epoch * args.traj_length, 3, 64, 64)
        if epoch == 1:
            with imageio.get_writer(args.log_dir + '/trial.gif', mode='I') as writer:
                for k, frame in enumerate(eps_obs[0]):
                    img = np.array(frame)
                    img = img.transpose((1, 2, 0)) * 255.0
                    writer.append_data(img.astype('uint8'))
        eps_next = np.array(eps_next).reshape(-1, args.num_traj_per_epoch * args.traj_length, 3, 64, 64)
        eps_act = np.array(eps_act).reshape(-1, args.num_traj_per_epoch * args.traj_length, img_buffer.action_dim)
        img_buffer.add_sample(
                    states=eps_obs,
                    next_states=eps_next,
                    actions=eps_act,
                    )
        
        # Gradually increase the horizon for training the dynamics model
        predlen = 10
        if epoch < 300:
            predlen = 8
        if epoch < 150:
            predlen = 4
        if epoch < 50:
            predlen = 2

        if epoch % args.update_freq == 0:
            print("Updating")
            if args.logging > 0 and epoch % args.loss_log_freq == 0:
                epoch_dynamics_loss = np.zeros((args.grad_steps_per_update,), dtype=float)
                epoch_vae_loss = np.zeros((args.grad_steps_per_update,), dtype=float)
                if args.use_classifiers is not None:
                    epoch_auxillary_loss = np.zeros((args.classifiers_grad_steps,), dtype=float)
                else:
                    epoch_auxillary_loss = np.zeros((args.grad_steps_per_update,), dtype=float)

            for grstep in range(args.grad_steps_per_update):
                losses = []
                # Return [batch_sz, predlen, 3, 64, 64]
                obs, next_obs, actions, success = img_buffer.draw_samples(batch_size=args.batch_sz, length=predlen)

                obs = torch.tensor(obs).float().to(device) 
                next_obs = torch.tensor(next_obs).float().to(device)
                actions = torch.tensor(actions).float().to(device)
                _, _, _, _, obs_z = enc_dec.forward(obs)
                _, _, _, _, next_z = enc_dec.forward(next_obs)
                g_ind = np.random.randint(0, len(goals), args.batch_sz) 
                g_samples = goals[g_ind]
                _, _, _, _, goal_z = enc_dec.forward(g_samples.unsqueeze(1))

                if args.dynamics_var:
                    ''' Train dynamics models in disagreement ensemble '''
                    dynamics_loss = train_disgrmt_ensemble(img_buffer, dynamics_models, enc_dec, dyn_optimizer, args.batch_sz, predlen)
                    auxillary_loss = dynamics_loss
                else:
                    dynamics_loss, pred_z = train_dynamics_model(dynamics_model, obs_z, next_z, actions, dyn_optimizer)
                if args.logging > 0 and epoch % args.model_log_freq == 0 and not args.dynamics_var:
                    dynamics_preds = enc_dec.dec(pred_z.float())
                    # decode pred_z through the decoder & compare with next_obs
                    for num in range(3):
                        dynamics_pred = dynamics_preds[num]
                        dynamics_true = next_obs[num]
                        dynamics_pred = ptu.get_numpy(dynamics_pred.permute(0, 2, 3, 1))
                        dynamics_true = ptu.get_numpy(dynamics_true.permute(0, 2, 3, 1))
                        dynamics_true = (dynamics_true * 255.).astype(np.uint8)
                        dynamics_pred = (dynamics_pred * 255.).astype(np.uint8)
                        path = args.log_dir + '/dynamics_preds/' + str(epoch)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        with imageio.get_writer(path + '/train_true' + str(num) + '.gif', mode='I') as writer:
                            for e in range(len(dynamics_true)):
                                writer.append_data(dynamics_true[e])
                        with imageio.get_writer(path + '/train_pred' + str(num) + '.gif', mode='I') as writer:
                            for e in range(len(dynamics_pred)):
                                writer.append_data(dynamics_pred[e])

                ''' Train classifiers ''' 
                if args.use_classifiers is not None and grstep < args.classifiers_grad_steps:
                    score_path = None
                    if args.logging > 0 and epoch % args.model_log_freq == 0:
                        score_path = args.log_dir + '/classifier_scores/' + str(epoch)
                        if not os.path.exists(score_path):
                            os.makedirs(score_path)
                    auxillary_loss = train_classifiers(classifiers, enc_dec, obs, goals, c_optimizer, args.batch_sz, score_path)
                
                ''' Update SMM density models '''
                if args.smm:
                    auxillary_loss = train_smm_density_models(density_vae, goal_vae, obs_z, goal_z, d_optimizer, g_optimizer)
                
                '''Train main vae'''
                vae_loss, g_rec, ng_rec = train_vae(args, enc_dec, obs, g_samples, enc_optimizer, args.beta)
                if args.logging > 0 and epoch % args.model_log_freq == 0:
                    # save g_rec & g_samples
                    if g_rec is not None:
                        g_rec = g_rec.cpu().detach()
                        g_rec = g_rec * 255.0
                        r_imgs = g_rec.squeeze(1).permute(0, 2, 3, 1).reshape(-1, 64, 64, 3)
                        r_imgs = ptu.get_numpy(r_imgs).astype(np.uint8)
                        g_true = g_samples * 255.0
                        t_imgs = g_true.permute(0, 2, 3, 1).reshape(-1, 64, 64, 3)
                        t_imgs = ptu.get_numpy(t_imgs).astype(np.uint8)
                        for im in range(5):
                            img = Image.fromarray(r_imgs[im])
                            path = args.log_dir + '/vae_recs/' + str(epoch)
                            if not os.path.exists(path):
                                os.makedirs(path)
                            img.save(path + '/g_rec' + str(im) + '.png')
                            img = Image.fromarray(t_imgs[im])
                            img.save(path + '/g_true' + str(im) + '.png')

                    ng_rec = ng_rec * 255.0
                    r_imgs = ng_rec.squeeze(1).permute(0, 2, 3, 1).reshape(-1, 64, 64, 3)
                    r_imgs = ptu.get_numpy(r_imgs).astype(np.uint8)
                    ng_true = obs[:,0,:,:,:] * 255.0
                    t_imgs = ng_true.permute(0, 2, 3, 1).reshape(-1, 64, 64, 3)
                    t_imgs = ptu.get_numpy(t_imgs).astype(np.uint8)
                    for im in range(5):
                        img = Image.fromarray(r_imgs[im])
                        path = args.log_dir + '/vae_recs/' + str(epoch)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        img.save(path + '/ng_rec' + str(im) + '.png')
                        img = Image.fromarray(t_imgs[im])
                        img.save(path + '/ng_true' + str(im) + '.png')

                if args.logging > 0 and epoch % args.loss_log_freq == 0:
                    epoch_dynamics_loss[grstep] = dynamics_loss
                    epoch_vae_loss[grstep] = vae_loss
                    if args.dynamics_var or args.smm or grstep < args.classifiers_grad_steps:
                        epoch_auxillary_loss[grstep] = auxillary_loss

        if args.logging > 0:
            end = time.time()
            print("===== EPISODE {} FINISHED IN {}s =====".format(epoch, end - start))

        if args.logging > 0 and epoch % args.loss_log_freq == 0 and epoch > 0:
            hist.save_losses(
                            epoch_auxillary_loss.mean(), # e.g. SMM, disagreement, classifiers max
                            epoch_dynamics_loss.mean(),
                            epoch_vae_loss.mean(),
                            )
            if args.logging == 2:
                print(hist.report_losses)
        
        if epoch % args.model_log_freq == 0:
            torch.save(enc_dec.state_dict(), enc_dec.savedir + '/{}model.bin'.format(epoch))
            if args.dynamics_var:
                for model in dynamics_models:
                    torch.save(model.state_dict(), model.savedir + '/{}model.bin'.format(epoch))
            else:
                torch.save(dynamics_model.state_dict(), dynamics_model.savedir + '/{}model.bin'.format(epoch))
            if args.use_classifiers is not None:
                for classifier in classifiers:
                    torch.save(classifier.state_dict(), classifier.savedir + '/{}model.bin'.format(epoch))
            if args.smm:
                torch.save(goal_vae.state_dict(), goal_vae.savedir + '/{}model.bin'.format(epoch))
                torch.save(density_vae.state_dict(), density_vae.savedir + '/{}model.bin'.format(epoch))
            if args.logging > 0:
                hist.save_losses_txt()

        if epoch == args.max_epoch:
            assert(False)


if __name__ == "__main__":
    import argparse
    args = parse_args()
    
    args = create_log_dir(args)
            
    with open(args.log_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    print("Log directory:", args.log_dir)
    import random
    random.seed(args.seed)
     
    main(args)
