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

        self.env = RobotEnv(
            robot_type="panda",
            action_space=cfg.action_space,
            gripper_action_space="position",
        )
        self.n_envs = 1

        self.n_cond_step = cfg.cond_steps
        self.n_img_cond_step = cfg.img_cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Eval params
        self.n_steps = cfg.n_steps

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)

    def run(self):
        pass

    def reset_env(self, randomize_reset=False):
        self.env.reset(randomize=randomize_reset)
        obs = self.env.get_observation()
        return obs
