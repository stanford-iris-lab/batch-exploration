from PIL import Image
import imageio
import numpy as np
import torch
import pytorch_util as ptu

import os

"""
Log reconstructed and simulated gifs of 5 top and bottom ranked trajectories
"""
def log_rankings(args, enc_dec, rankings_filepath, viz_env, init_low_dim, sample_actions, ordered_preds, epoch, traj, action_rewards):
    rankings_save_epoch = rankings_filepath + '/{}'.format(epoch)
    if not os.path.exists(rankings_save_epoch):
        os.mkdir(rankings_save_epoch)

    rankings_save_traj = rankings_save_epoch + '/{}'.format(traj)
    if not os.path.exists(rankings_save_traj):
        os.mkdir(rankings_save_traj)

    for j in range(5):
        if viz_env is not None:
            imgs = viz_env.take_steps_and_render(init_low_dim, sample_actions[j])
            with imageio.get_writer(rankings_save_traj + '/{}sim.gif'.format(j), mode='I') as writer:
                for e in range(10):
                    writer.append_data(imgs[e].astype(np.uint8))
        '''Save VAE reconsructed gif.'''
        ordered_pred = torch.tensor(ordered_preds[j]).cuda()
        recons = enc_dec.dec(ordered_pred.unsqueeze(1).float())
        recons = recons * 255.0
        imgs = ptu.get_numpy(recons.squeeze(1).permute(0, 2, 3, 1)).reshape(-1, 64, 64, 3).astype(np.uint8)
        with imageio.get_writer(rankings_save_traj + '/{}vae.gif'.format(j), mode='I') as writer:
            for e in range(10):
                writer.append_data(imgs[e])

    for j in range(999, 994, -1):
        if viz_env is not None:
            imgs = viz_env.take_steps_and_render(init_low_dim, sample_actions[j])
            with imageio.get_writer(rankings_save_traj + '/{}sim.gif'.format(j), mode='I') as writer:
                for e in range(10):
                    writer.append_data(imgs[e].astype(np.uint8))
        '''Save VAE reconsructed gif'''
        ordered_pred = torch.tensor(ordered_preds[j]).cuda()
        recons = enc_dec.dec(ordered_pred.unsqueeze(1).float()) 
        recons = recons * 255.0
        imgs = ptu.get_numpy(recons.squeeze(1).permute(0, 2, 3, 1)).reshape(-1, 64, 64, 3).astype(np.uint8) 
        with imageio.get_writer(rankings_save_traj + '/{}vae.gif'.format(j), mode='I') as writer:
            for e in range(10):
                writer.append_data(imgs[e])
                
    np.savetxt(rankings_save_traj + '/action_rewards.txt', np.array(action_rewards), fmt='%f')
