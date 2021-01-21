from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
# import torch
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from PIL import Image
from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import quat_mul, quat2axisangle
import cv2
import imageio
import time
import inspect
import sys
import mujoco_py

from manip_env.utils import _door_goal, _drawer_goal, _block_goal 

class Tabletop(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            goal_low=None,
            goal_high=None,
            hand_init_pos=(0, 0.6, 0.2),
            liftThresh=0.04,
            rewMode='orig',
            rotMode='rotz',
            problem="rand",
            door=0, # 0: not using door env, 1: using default door, 3: door w/ 3 distractors, 5: door w/ 5 distractors
            drawer=False,
            drawers=False,
            exploration = "hard",
            filepath="test",
            max_path_length=50,
            verbose=1,
            hard=False,
            log_freq=100, # in terms of episode num
            **kwargs
    ):
        self.randomize = False
        self.door = door # non zero, use door env (w/ varying distractors)
        self.hard = hard # if True, blocks are initialized to diff corners
        self.drawer = drawer
        self.drawers = drawers
        self.exploration = exploration
        self.max_path_length = max_path_length
        self.cur_path_length = 0
        self.quick_init(locals())
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.20)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./20,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.liftThresh = liftThresh
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.action_rot_scale = 1./10
        self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.imsize = 64
        self.goal_space = self.observation_space
        
        '''For Logging'''
        self.verbose = verbose
        if self.verbose:
            self.imgs = []
            self.filepath = filepath
            if not os.path.exists(self.filepath):
                os.mkdir(self.filepath)
        self.log_freq = log_freq
        self.epcount = 0 # num episodes so far 
        self.good_qpos = None # self.data.qpos[:7]

    @property
    def observation_space(self):
        return Box(0, 1.0, (self.imsize*self.imsize*3, ))

    @property
    def model_name(self):
        dirname = os.path.dirname(__file__)
        if self.exploration == "easy":
            filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject.xml") # three easy blocks
        else:
            if self.door == 1:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_door_3_blocks.xml") # three stacked blocks plus door
            elif self.door == 3:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_door_3_towers.xml") # three tall blocks spread out plus door
            elif self.door == 5:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_door_5_towers.xml") # three tall blocks spread out plus door
            elif self.drawer:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_cluttered_drawer.xml") #
            elif self.drawers:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_drawers.xml") # cluttered drawer
            else:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_hard.xml") # three blocks but spread out
        return filename

    def _get_low_dim_info(self):
        if not (self.door or self.drawer or self.drawers):
            env_info =  {'block0_x': self.data.qpos[9], 
                        'block0_y': self.data.qpos[10], 
                        'block0_z': self.data.qpos[11], 
                        'block1_x': self.data.qpos[12], 
                        'block1_y': self.data.qpos[13], 
                        'block1_z': self.data.qpos[14], 
                        'block2_x': self.data.qpos[15], 
                        'block2_y': self.data.qpos[16], 
                        'block2_z': self.data.qpos[17], 
                        'hand_x': self.get_endeff_pos()[0],
                        'hand_y': self.get_endeff_pos()[1],
                        'hand_z': self.get_endeff_pos()[2],
                        'dist': - self.compute_reward()}
        elif self.drawers:
            env_info = {'hand_x': self.get_endeff_pos()[0],
                        'hand_y': self.get_endeff_pos()[1],
                        'hand_z': self.get_endeff_pos()[2],
                        'dist': - self.compute_reward()}
        else:
            env_info =  {'block0_x': self.data.qpos[9], 
                        'block0_y': self.data.qpos[10], 
                        'block0_z': self.data.qpos[11], 
                        'block1_x': self.data.qpos[16], 
                        'block1_y': self.data.qpos[17], 
                        'block1_z': self.data.qpos[18], 
                        'block2_x': self.data.qpos[23], 
                        'block2_y': self.data.qpos[24], 
                        'block2_z': self.data.qpos[25], 
                        'hand_x': self.get_endeff_pos()[0],
                        'hand_y': self.get_endeff_pos()[1],
                        'hand_z': self.get_endeff_pos()[2],
                        'dist': - self.compute_reward()}
            if self.door:
                env_info['door'] = self.data.qpos[-1]
        if self.door == 5:
            env_info['block3_x'] = self.data.qpos[30]
            env_info['block3_y'] = self.data.qpos[31]
            env_info['block3_z'] = self.data.qpos[32]
            env_info['block4_x'] = self.data.qpos[37]
            env_info['block4_y'] = self.data.qpos[38]
            env_info['block4_z'] = self.data.qpos[39] 
        if self.drawer:
            env_info['block3_x'] = self.data.qpos[30]
            env_info['block3_y'] = self.data.qpos[31] 
            env_info['block3_z'] = self.data.qpos[32] 
            env_info['block4_x'] = self.data.qpos[37] 
            env_info['block4_y'] = self.data.qpos[38] 
            env_info['block4_z'] = self.data.qpos[39] 
            env_info['block5_x'] = self.data.qpos[44] 
            env_info['block5_y'] =  self.data.qpos[45] 
            env_info['block5_z'] = self.data.qpos[46] 
            env_info['drawer'] = self.data.qpos[-1]
        elif self.drawers:
            env_info['drawer'] = self.data.qpos[-1]
        return env_info


    def step(self, action):
        self.set_xyz_action_rotz(action[:4])
        self.do_simulation([action[-1], -action[-1]])

        ob = None
        ob = self.get_obs()
        reward  = self.compute_reward()
        if self.cur_path_length == self.max_path_length:
            done = True
        else:
            done = False
        
        '''
        For logging
        Render images from every step if saving current episode
        '''
        if self.verbose:
            if self.epcount % self.log_freq == 0:
                im = self.sim.render(self.imsize, self.imsize, camera_name='cam0')
                self.imgs.append(im)

        self.cur_path_length +=1
        low_dim_info = self._get_low_dim_info()
        return ob, reward, done, low_dim_info
   
    def get_obs(self):
        obs = self.sim.render(self.imsize, self.imsize, camera_name="cam0") / 255.
        return obs
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        if self.door:
            start_id = 9 + self.targetobj*7
        elif self.drawer:
            start_id = 9 + self.targetobj*7
        elif self.drawers:
            start_id = 9 + self.targetobj
        else:
            start_id = 9 + self.targetobj*3
        if len(pos) < 3:
            qpos[start_id:(start_id+2)] = pos.copy()
            qvel[start_id:(start_id+2)] = 0
        else:
            qpos[start_id:(start_id+3)] = pos.copy()
            qvel[start_id:(start_id+3)] = 0
        self.set_state(qpos, qvel)
          
    def initialize(self):
        self.epcount = -1 # to ensure the first episode starts with 0 idx
        self.cur_path_length = 0

    def reset_model(self, add_noise=False, just_restore=False):
        ''' For logging '''
        if self.verbose and not just_restore:
            if self.epcount % self.log_freq == 0:
                self.save_gif() # save episode gif
        self.cur_path_length = 0
        if not just_restore:
            self.epcount += 1
            
        self._reset_hand()
        for _ in range(100):
            self.do_simulation([0.0, 0.0])
        self.targetobj = np.random.randint(3)
        self.cur_path_length = 0
        obj_num = 3
        if self.door == 5:
            obj_num = 5
        elif self.drawer:
            obj_num = 6
        elif self.drawers:
            obj_num = 0
        for i in range(obj_num):
            self.targetobj = i
            if self.randomize:
                init_pos = np.random.uniform(
                -0.2,
                0.2,
                size=(2,),
            )
            elif self.hard:
                if i == 0:
                    init_pos = [-.2, 0]
                elif i == 1:
                    init_pos = [-.1, .15]
                else:
                    init_pos = [ .2, -.1]
            else:
                init_pos = [0.1 * (i-1), 0.15] 
            if self.door:
                if self.door == 1:
                    init_pos = [-0.15, 0.75, 0.05 * (i+1)]
                    init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                elif self.door == 3:
                    if i == 0:
                        init_pos = [-0.15, 0.8, 0.075]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.075]
                    if i == 2:
                        init_pos = [0.25, 0.4, 0.075]
                    init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                elif self.door == 5:
                    if i == 0:
                        init_pos = [-0.15, 0.8, 0.075]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.075]
                    if i == 2:
                        init_pos = [0.25, 0.4, 0.075]
                    if i == 3:
                        init_pos = [0.25, 0.6, 0.075]
                    if i == 4:
                        init_pos = [0.15, 0.6, 0.075]
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            elif self.drawer or self.drawers:
                if i == 0:
                    init_pos = [0.35, 0.3, 0.05]
                if i == 1:
                    init_pos = [-0.12, 0.6, 0.05]
                if i == 2:
                    init_pos = [0.2, 0.3, 0.05]
                if i == 3:
                    init_pos = [-0.15, 0.4, 0.05]
                if i == 4:
                    init_pos = [0.45, 0.6, 0.05]
                if i == 5:
                    init_pos = [-0.2, 0.7, 0.05]
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            if add_noise:
                init_pos += np.random.uniform(-0.02, 0.02, (2,))

            self.obj_init_pos = init_pos
            self._set_obj_xyz(self.obj_init_pos)
            # tower pos needs to be initialized via set_joint_qpos
            if self.door or self.drawer or self.drawers:
                object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                object_qpos[:3 ] = init_pos
                object_qpos[3:] = 0.
                self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
        
        if self.door:
            self.data.qpos[-1] = 0.
        elif self.drawer or self.drawers:
            self.data.qpos[-1] = -0.05
        self.sim.forward()
        o = self.get_obs()
        
        if self.epcount % self.log_freq == 0 and not just_restore:
            self.imgs = []
            im = self.sim.render(self.imsize, self.imsize, camera_name='cam0')
            self.imgs.append(im)
        low_dim_info = self._get_low_dim_info()
        return o, low_dim_info 


    def _reset_hand(self, pos=None):
        if self.epcount < 10 and self.cur_path_length == 0:
            self.good_qpos = self.sim.data.qpos[:7].copy()
        self.data.qpos[:7] = self.good_qpos
        if pos is None:
            pos = self.hand_init_pos.copy()
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        return 0.0
    
    def get_goal(self, block, fixed_angle=None):
        ''' Returns a random goal img depending on the desired block/door '''
        goal_pos = None
        angle = 0.
        if self.door:
            # If want to set door as the target, uncomment below
            # hinge between +/- 45 degrees, at least abs > 20 degrees
            while abs(angle) < 0.0872665:# larger than 5 degrees angle
                angle = np.random.uniform(-0.785398, 0.785398)
            if fixed_angle is not None:
                angle = fixed_angle
            if self.door == 5:
                blocks_pos = _door_goal(block_num=5)
                gripper_pos = self.sim.data.get_geom_xpos('handle')
                self.data.qpos[-1] = angle
                goal_pos = np.concatenate([gripper_pos, blocks_pos])
            elif self.door == 1:
                blocks_pos = np.concatenate([self.data.qpos[9:12], self.data.qpos[16:19], self.data.qpos[23:26]])
            elif self.door == 3:
                blocks_pos = _door_goal(block_num=3)
            gripper_pos = self.sim.data.get_geom_xpos('handle')
            self.data.qpos[-1] = angle
            if self.door != 5:
                goal_pos = np.concatenate([gripper_pos, blocks_pos])
        elif self.drawer or self.drawers:
            # slightly increased the goal range from (0, 0.2) to below
            angle = np.random.uniform(0.05, 0.14)
            angle = -angle
            if fixed_angle is not None:
                angle = fixed_angle
            blocks_pos = _drawer_goal()
            self.data.qpos[-1] = angle
            gripper_pos = self.data.get_site_xpos('handleStart')
            goal_pos = np.concatenate([gripper_pos, blocks_pos])
        elif block == 0: # green block
            goal_pos = _block_goal(block_num=0, hard=self.hard) 
        elif block == 1: # pink block
            goal_pos = _block_goal(block_num=1, hard=self.hard)
        elif block == 2: # blue block
            goal_pos = _block_goal(block_num=2, hard=self.hard) 
        if self.door or self.drawer or self.drawers:
            goal_img = self.save_goal_img(goal_pos, angle=angle)
        else:
            goal_img = self.save_goal_img(goal_pos)
        return goal_img
    
    def take_steps_and_render(self, obs, actions, set_qpos=None):
        '''Returns image after having taken actions from obs.'''
        threshold = 0.05
        repeat = True
        _iters = 0
        if set_qpos is not None:
            self.data.qpos[:] = set_qpos.copy()
        else:
            self.reset_model()
            while repeat:
                obj_num = 3
                if self.door == 5:
                    obj_num = 5
                elif self.drawer or self.drawers:
                    obj_num = 6
                for i in range(obj_num):
                    self.targetobj = i
                    if self.door or self.drawer or self.drawers:
                        # init_pos = obs[(i+1)*3:((i+1)*3)+3]
                        self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+3]
                        self._set_obj_xyz(self.obj_init_pos)
                    else:
                        self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+2]
                        self._set_obj_xyz(self.obj_init_pos)
                if not (self.door or self.drawer) or self.drawers:
                    error = np.linalg.norm(obs[3:12] - self.data.qpos[9:18])
                    repeat = (error >= threshold)
                    _iters += 1
                else:
                    break
                if _iters > 10:
                    break
            repeat = True
            _iters = 0
            if self.door: 
                self.data.qpos[-1] = obs[-1]
                door_vel = np.array([0.])
                self.sim.data.set_joint_qvel('doorjoint', door_vel)
            elif self.drawer or self.drawers:
                self.data.qpos[-1] = obs[-1]
        self._reset_hand(pos=obs[:3])
        imgs = []
        im = self.sim.render(64, 64, camera_name='cam0')
        imgs.append(im)
        ''' Then take the selected actions '''
        for i in range(actions.shape[0]):
            action = actions[i]
            self.set_xyz_action_rotz(action[:4])
            self.do_simulation([action[-1], -action[-1]])
            im = self.sim.render(64, 64, camera_name='cam0')
            imgs.append(im)
        return imgs
        
    def _restore(self):
        '''For resetting the env without having to call reset() (i.e. without updating episode count)'''
        self.reset_model(just_restore=True)

    def save_goal_img(self, goal, actions=None, angle=None):
        '''Returns image with a given goal array of positions for the gripper and blocks.'''
        if self.drawer or self.drawers:
            goal[:3] = self.data.get_site_xpos('handleStart')
        pos = goal[:3]
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

        #  Move blocks to correct positions
        obj_num = 3
        if self.door == 5:
            obj_num = 5
        elif self.drawer:
            obj_num = 6
        elif self.drawers:
            obj_num = 0
        for i in range(obj_num):
            self.targetobj = i
            init_pos = None
            if self.door or self.drawer or self.drawers:
                init_pos = goal[(i+1)*3:((i+1)*3)+3]
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+3]
            else:
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+2]
            self.obj_init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                
            self._set_obj_xyz(self.obj_init_pos)
          
            if self.door or self.drawer:
                object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                object_qpos[:3] = init_pos
                object_qpos[3:] = 0.
                self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            self.sim.forward()
        
        if angle is not None:
            self.data.qpos[-1] = angle
        im = self.sim.render(64, 64, camera_name='cam0')
        return im

    
    def save_gif(self):
        ''' Saves the gif of an episode '''
        with imageio.get_writer(
                self.filepath + '/Eps' + str(self.epcount) + '.gif', mode='I') as writer:
            for i in range(self.max_path_length + 1):
                writer.append_data(self.imgs[i])
                
    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def set_xyz_action_rotz(self, action):
        self.set_xyz_action(action[:3])
        zangle_delta = action[3] * self.action_rot_scale
        new_mocap_zangle = quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta

        # new_mocap_zangle = action[3]
        new_mocap_zangle = np.clip(
            new_mocap_zangle,
            -3.0,
            3.0,
        )
        if new_mocap_zangle < 0:
            new_mocap_zangle += 2 * np.pi
        self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))


def quat_to_zangle(quat):
    q = quat_mul(quat_inv(quat_create(np.array([0, 1., 0]), np.pi / 2)), quat)
    ax, angle = quat2axisangle(q)
    return angle


def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return quat_mul(quat_create(np.array([0, 1., 0]), np.pi / 2),
                    quat_create(np.array([-1., 0, 0]), zangle))


def quat_create(axis, angle):
    """
        Create a quaternion from an axis and angle.
        :param axis The three dimensional axis
        :param angle The angle in radians
        :return: A 4-d array containing the components of a quaternion.
    """
    quat = np.zeros([4], dtype='float')
    mujoco_py.functions.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def quat_inv(quat):
    """
        Invert a quaternion, represented by a 4d array.
        :param A quaternion (4-d array). Must not be the zero quaternion (all elements equal to zero)
        :return: A 4-d array containing the components of a quaternion.
    """
    d = 1. / np.sum(quat ** 2)
    return d * np.array([1., -1., -1., -1.]) * quat