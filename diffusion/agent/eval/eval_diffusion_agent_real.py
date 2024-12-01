"""
Parent eval agent class.

"""

import os
import numpy as np
import torch
import hydra
import logging
import random

from droid.robot_env import RobotEnv

log = logging.getLogger(__name__)

from multiprocessing.managers import SharedMemoryManager


class EvalDiffusionAgentReal:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # shm_manager = SharedMemoryManager()

        self.n_cond_step = cfg.cond_steps
        self.n_img_cond_step = cfg.img_cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.debug = cfg.get("eval_debug", False)

        # Initialize environment

        if self.debug:
            self.env = DummyEnv(
                img_size=cfg.shape_meta.obs.rgb.shape
            )
        else:
            self.env = RobotEnv(
                robot_type="panda",
                action_space=cfg.action_space,
                gripper_action_space="position",
                camera_resolution=(cfg.shape_meta.obs.rgb.shape[1], cfg.shape_meta.obs.rgb.shape[0]),   # TODO: resolutions for different cameras
            )
        
        self.n_envs = 1

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Eval params
        self.n_steps = cfg.n_steps

        # Logging
        self.logdir = cfg.logdir
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.logdir, exist_ok=True)

    def run(self):
        pass

    def reset_env(self, randomize_reset=False):
        self.env.reset(randomize=randomize_reset)
        obs = self.env.get_observation()
        return obs
    
class DummyEnv:
    def __init__(self, img_size):

        self.img_size = img_size
        assert len(self.img_size) == 3 and self.img_size[0] == 3
        self.img_size = (self.img_size[1], self.img_size[2], self.img_size[0])
        class CameraReader:
            def set_trajectory_mode(self):
                pass
        self.camera_reader = CameraReader()

    def reset(self, randomize):
        pass
    
    def get_observation(self):
        img = {f'{i}': np.random.randint(0, 255, size=(self.img_size)) for i in [0,3]}
        obs = {'image': img, 'robot_state': {'cartesian_position': np.random.rand(6), 'joint_positions': np.random.rand(7), 'gripper_position': np.random.rand(1).item()}}
        return obs

    def step(self, action):
        pass