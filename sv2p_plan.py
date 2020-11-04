from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import flags
import time

from manip_envs.tabletop import Tabletop
from utils.utils import get_obs
import cv2
import numpy as np
from PIL import Image
import imageio
from tensor2tensor.bin.t2t_decoder import create_hparams
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import argparse
import json

FLAGS = flags.FLAGS

def save_im(im, name):
    img = Image.fromarray(im.astype(np.uint8))
    img.save(name)

 
class CEM(object):
    def __init__(self, savedir, phorizon,
               cem_samples, cem_iters, cost='pixel'):
        
        self.eps = 0
        self.savedir = savedir
        self.planstep = 0
        self.phorizon = phorizon
        self.cem_samples = cem_samples
        self.cem_iters = cem_iters
        self.verbose = False
        self.num_acts = 5
        self.cost = cost

        # LOADING SV2P 
        FLAGS.data_dir = args.root + 'data/'

        FLAGS.problem = args.problem
        FLAGS.hparams = 'video_num_input_frames=5,video_num_target_frames=15'
        FLAGS.hparams_set = 'next_frame_sv2p'
        FLAGS.model = 'next_frame_sv2p'
        # Create hparams
        hparams = create_hparams()
        hparams.video_num_input_frames = 1
        hparams.video_num_target_frames = self.phorizon

        # Params
        num_replicas = self.cem_samples
        frame_shape = hparams.problem.frame_shape
        forward_graph = tf.Graph()
        with forward_graph.as_default():
            self.forward_sess = tf.Session()
            input_size = [num_replicas, hparams.video_num_input_frames]
            target_size = [num_replicas, hparams.video_num_target_frames]
            self.forward_placeholders = {
              'inputs':
                  tf.placeholder(tf.float32, input_size + frame_shape),
              'input_action':
                  tf.placeholder(tf.float32, input_size + [self.num_acts]),
              'targets':
                  tf.placeholder(tf.float32, target_size + frame_shape),
              'target_action':
                  tf.placeholder(tf.float32, target_size + [self.num_acts]),
            }
            # Create model
            forward_model_cls = registry.model(FLAGS.model)
            forward_model = forward_model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
            self.forward_prediction_ops, _ = forward_model(self.forward_placeholders)
            forward_saver = tf.train.Saver()
            forward_saver.restore(self.forward_sess,
                                args.model_dir)
        print('LOADED SV2P!')


    def cem(self, forward_sess, forward_placeholders, forward_ops, curr, goal, env, eps, planstep, verbose):
        """Runs Visual MPC between two images."""
        horizon = forward_placeholders['targets'].shape[1]
        mu1 = np.array([0]*(self.num_acts * horizon))
        sd1 = np.array([0.2]*(self.num_acts * horizon))

        _iter = 0
        sample_size = self.cem_samples 
        resample_size = self.cem_samples // 5
        
        hz = horizon

        while np.max(sd1) > .001:
            if _iter == 0:
                acts1 = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[sample_size, hz, self.num_acts])
            else:
                acts1 = np.random.normal(mu1, sd1, (sample_size, self.num_acts *  hz))
            acts1 = acts1.reshape((sample_size, hz, self.num_acts))
            acts0 = acts1[:, 0:1, :]

            forward_feed = {
              forward_placeholders['inputs']:
                  np.repeat(np.expand_dims(np.expand_dims(curr, 0), 0),
                            sample_size, axis=0),
              forward_placeholders['input_action']:
                  acts0,
              forward_placeholders['targets']:
                  np.zeros(forward_placeholders['targets'].shape),
              forward_placeholders['target_action']:
                  acts1
            }
            forward_predictions = forward_sess.run(forward_ops, forward_feed)

            if self.cost == 'temporal':
                losses = self.temporal_cost(forward_predictions.reshape(-1, 64, 64, 3), goal)
                losses = losses.reshape(sample_size, horizon)
            elif self.cost == 'pixel':
                goalim = np.repeat(np.expand_dims(
                    np.repeat(np.expand_dims(goal, 0), horizon, 0), 0), sample_size, 0)
                losses = (goalim - forward_predictions.squeeze(-1))**2
                losses = losses.mean(axis=(1, 2, 3, 4)) # average cost between goal and every pred in the traj
            best_actions = np.array([x for _, x in sorted(
              zip(losses, acts1.tolist()), reverse=False)])
            best_costs = np.array([x for x, _ in sorted(
              zip(losses, acts1.tolist()), reverse=False)])

            """ Log top 10 and bottom 10 trajs """
            if verbose > 0 and _iter == 0:
                forward_predictions = forward_predictions.squeeze(-1)
                for q in range(10):
                    head = self.savedir + 'rankings/{}/{}/'.format(eps, planstep)
                    if not os.path.exists(head):
                        os.makedirs(head)
                    if q == 0:
                        save_im(curr, head+'curr.jpg')
                        save_im(goal, head+'goal.jpg')
                    with imageio.get_writer('{}pred{}_{}.gif'.format(head, q, best_costs[q]), mode='I') as writer:
                        for p in range(horizon):
                            writer.append_data(forward_predictions[q, p, :, :, :].astype('uint8'))
                for q in range(args.cem_samples-1, args.cem_samples-11, -1):
                    head = self.savedir + 'rankings/{}/{}/'.format(eps, planstep)
                    with imageio.get_writer('{}pred{}_{}.gif'.format(head, q, best_costs[q]), mode='I') as writer:
                        for p in range(horizon):
                            writer.append_data(forward_predictions[q, p, :, :, :].astype('uint8'))
            best_actions = best_actions[:resample_size]
            best_costs = best_costs[:resample_size]
            best_actions1 = best_actions.reshape(resample_size, -1)
            mu1 = np.mean(best_actions1, axis=0)
            sd1 = np.std(best_actions1, axis=0)
            _iter += 1
            if _iter >= self.cem_iters:
                break

        chosen = best_actions1[0]
        bestcost = best_costs[0]
        return chosen, bestcost


def get_goal_img(args, env):
    _, env_info = env.reset_model()
    fixed_angle = None
    passing_angle= None
    if args.door:
        if args.task == 0:
            fixed_angle = 0.34907
            passing_angle = np.array([-0.174533, 0.174533]) + fixed_angle
        elif args.task == 1:
            fixed_angle = 1.5708
            passing_angle = fixed_angle
        elif args.task == 2:
            # target angle between 10 and 30 degrees
            fixed_angle = np.random.uniform(0.349066, 0.785398)
            passing_angle = np.array([-0.174533, 0.174533]) + fixed_angle
        print("Goal: {} | Pass: {}".format(fixed_angle * 180. / np.pi, passing_angle * 180. /np.pi))
        obs, env_info = env.reset_model()
        goal_pos = get_obs(args, env_info)
        goal_pos[-1] = fixed_angle

    elif args.drawer: # Get goal for drawer 
        fixed_angle = -0.12
        goal_position = fixed_angle
        print("Goal: ", fixed_angle)
        obs, env_info = env.reset_model()
        goal_pos = get_obs(args, env_info)
        goal_pos[-1] = fixed_angle
            
    else: # Get goal for blocks
        curr_pos = get_obs(args, env_info)
        if args.goal_block == 0:
            if args.fixed: # test on fixed target for block 0
                goal_block_0_pos = np.array([-.2, 0.2, 0])
            else:
                goal_block_0_pos[:2] = np.random.uniform( # green block
                            (-0.2, -0.2),  
                            (-0.1, -0.1), 
                            size=(2,))
                goal_block_0_pos[2] = 0.02
            gripper_pos = curr_pos[3:6].copy()
            gripper_pos[2] = 0.02
            gripper_pos[1] += 0.58 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
            curr_pos[:3] = gripper_pos
            curr_pos[3:6] = goal_block_0_pos
            goal_pos = curr_pos

        if args.goal_block == 1:
            # Push pink to bottom, randomly left or right +/-0.1 refit to 10
            if args.fixed: # test on fixed target for block 0
                goal_block_1_pos = np.array([-.26, 0.27, 0]) # -.26, 0.27, 0
            else: # get random target pos for block 0
                goal_block_1_pos = np.random.uniform( # green block
                            (-0.28, 0.23, 0),  
                            (-0.20, 0.30, 0), 
                            size=(3,))
                goal_block_1_pos[2] = 0
            gripper_pos = curr_pos[6:9].copy() #goal_block_1_pos.copy()
            gripper_pos[2] = 0.02
            gripper_pos[1] += 0.6 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
            curr_pos[:3] = gripper_pos
            curr_pos[6:9] = goal_block_1_pos
            goal_pos = curr_pos
                
        if args.goal_block == 2:
            if args.fixed: # test on fixed target for block 2
                goal_block_2_pos = np.array([0.3, -0.1, 0])
            else: # get random target pos for block 0
                goal_block_2_pos[:2] = np.random.uniform( # green block
                            (-0.2, -0.2),  
                            (-0.1, -0.1), 
                            size=(2,))
                goal_block_2_pos[2] = 0.02
            gripper_pos = curr_pos[-3:].copy() #goal_block_2_pos.copy()
            gripper_pos[2] = 0.02
            gripper_pos[1] -= 0.0
            gripper_pos[1] += 0.55 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
            curr_pos[:3] = gripper_pos
            curr_pos[-3:] = goal_block_2_pos
            goal_pos = curr_pos
    goal = env.get_goal(0, fixed_angle=fixed_angle)
    if args.door or args.drawer:
        goal_pos[:3] = env.get_endeff_pos()[:3]
    return goal, goal_pos


def main(args):
    # Load in models and env
    print("----Load in models and env----")
    sv2p = CEM(savedir=args.savedir, phorizon=args.phorizon,
               cem_samples = args.cem_samples, cem_iters = args.cem_iters)
    
    env = Tabletop(
            log_freq=50, 
            filepath=args.savedir,
            hard=True,
            door=args.door,
            drawer=args.drawer,
            verbose=False)
    
    print("----Done loading in models and env----")
    path = 5 # num of trajs per episode
    hz = 10 # traj_length
    full_low_dim = []
    goal_low_dim = []

    env.initialize()
    for eps in range(args.num_eps):
        eps_low_dim = []
        start = time.time()
        env.reset_model()
        ''' Get goal position for the trial '''
        if (not args.fixed) or eps < 1:
            goal, goal_pos = get_goal_img(args, env)
            im = Image.fromarray(goal.astype(np.uint8))
            im.save(args.savedir + '/goal{}.png'.format(eps))
        goal_low_dim.append(goal_pos)
        obs, env_info = env.reset_model()
        init_im = env.get_obs() * 255 
        if eps == 0 and args.verbose:
            save_im(init_im, '{}/init.png'.format(args.savedir))

        step = 0
        resample = -1 
        low_dim_state = get_obs(args, env_info)
        print("state dim: ", len(low_dim_state))
        eps_low_dim.append(low_dim_state)
        while step < path: # each episode is 5 x 10-step trajectories
            resample += 1
            if step == 0:
                obs = obs * 255
            chosen, bestcost = sv2p.cem(sv2p.forward_sess, sv2p.forward_placeholders, 
                                        sv2p.forward_prediction_ops, obs, goal, 
                                        env, eps, step, args.verbose)
            for h in range(hz): 
                obs, _, done, env_info = env.step(chosen[5*h:5*h+5]) # Step the environoment with the best action 
                obs = obs * 255
                if args.verbose:
                    save_im(obs, '{}step{}.png'.format(args.savedir, step * hz + h))
                low_dim_state = get_obs(args, env_info)
                eps_low_dim.append(low_dim_state)
            step += 1

        if args.verbose:
            with imageio.get_writer('{}{}.gif'.format(args.savedir, eps), mode='I') as writer:
                writer.append_data(imageio.imread('{}init.png'.format(args.savedir)))
                for step in range(path * 10):
                    img_path = '{}step{}.png'.format(args.savedir, step)
                    writer.append_data(imageio.imread(img_path))

        full_low_dim.append(np.array(eps_low_dim))
        end = time.time()
        print("Time for 1 trial", end - start)
        # print("===================================")
        print("-----------------EPS {} ENDED--------------------".format(eps))
    import pickle
    pickle.dump(np.array(full_low_dim), open(args.savedir + 'full_states.p', 'wb'))
    pickle.dump(np.array(goal_low_dim), open(args.savedir + 'goal_states.p', 'wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Change depending on the exp, just need to set the rootand savedir
    parser.add_argument("--root") # dir to tensor2tensor model
    parser.add_argument("--savedir", default='exps/') # planning dir for saving
    parser.add_argument("--problem", default='batch_exploration_block2_max') 
    parser.add_argument("--door", type=int, default=5)
    parser.add_argument("--drawer", default=False, action='store_true')
    parser.add_argument("--fixed", default=True, action='store_true')
    parser.add_argument("--verbose", default=0, type=int)
    
    parser.add_argument("--num_eps", type=int, default=1000) #100
    # task 0: fixed angle, task 1: fully open (60 degrees); evalaute on final dist to goal, task 2: btwn 10~30 degrees
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--cem_samples", type=int, default=200) #200
    parser.add_argument("--cem_iters", type=int, default=2) #5
    parser.add_argument("--phorizon", type=int, default=10) #planning horizon

    args = parser.parse_args()
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    args.model_dir = args.root + 'out/model.ckpt-200000'
        
    with open(args.savedir + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    main(args)
