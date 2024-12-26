"""
Parent eval agent class.

"""

import logging
import os
import random

import hydra
import numpy as np
import torch

from guided_dc.utils.pose_utils import quaternion_to_euler_xyz

log = logging.getLogger(__name__)


def process_obs(obs, obs_min, obs_max, ordered_obs_keys):
    """concatenate and normalize obs"""
    processed_obs = []
    for i in range(len(ordered_obs_keys)):
        o = obs["robot_state"][ordered_obs_keys[i]]
        o = np.array(o)
        if len(o.shape) == 1:
            o = o.reshape(1, -1)
        elif len(o.shape) == 0:
            o = o.reshape(1, 1)
        assert len(o.shape) == 2, o.shape
        processed_obs.append(o)

    processed_obs = np.concatenate(processed_obs, axis=1)
    processed_obs = 2 * (processed_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
    processed_obs = np.clip(processed_obs, -1, 1)
    # If any of the processed_obs reach the limits, print a warning and show which obs reached the limits
    # if np.any(np.abs(processed_obs[:, :-1]) == 1):
    #     raise ValueError(
    #         f"obs reached limits, {np.where(np.abs(processed_obs[:, :-1]) == 1)}"
    #     )

    # processed_obs[0,3] = np.abs(processed_obs[0,3])
    return processed_obs


class EvalAgent:
    def __init__(self, cfg, env=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # shm_manager = SharedMemoryManager()

        # Set obs dim -  we will save the different obs in batch in a dict
        self.ordered_obs_keys = cfg.ordered_obs_keys
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
        self.normalization_stats_path = cfg.normalization_stats_path
        self.normalization_stats = np.load(
            self.normalization_stats_path, allow_pickle=True
        )
        self.obs_min = self.normalization_stats["obs_min"]
        self.obs_max = self.normalization_stats["obs_max"]
        self.action_min = self.normalization_stats["action_min"]
        self.action_max = self.normalization_stats["action_max"]

        self.n_cond_step = cfg.cond_steps
        self.n_img_cond_step = cfg.img_cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.horizon_steps = cfg.horizon_steps
        self.act_steps = min(cfg.act_steps, self.horizon_steps)
        self.use_delta_actions = cfg.get("use_delta_actions", False)
        if self.use_delta_actions:
            self.delta_min = self.normalization_stats["delta_min"].astype(np.float32)
            self.delta_max = self.normalization_stats["delta_max"].astype(np.float32)
        self.debug = cfg.get("eval_debug", False)

        # Initialize environment
        if env is None:
            if self.debug:
                from guided_dc.envs.dummy_env import DummyEnv

                self.env = DummyEnv(img_size=cfg.shape_meta.obs.rgb.shape)
            else:
                from droid.robot_env import RobotEnv

                self.env = RobotEnv(
                    robot_type="panda",
                    action_space=cfg.action_space,
                    gripper_action_space="position",
                    camera_resolution=(
                        cfg.shape_meta.obs.rgb.shape[2],
                        cfg.shape_meta.obs.rgb.shape[1],
                    ),  # TODO: resolutions for different cameras
                )
        else:
            self.env = env

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Eval params
        self.n_steps = cfg.n_steps

        # Logging
        self.logdir = cfg.logdir
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.logdir, exist_ok=True)

    def run(self):
        raise NotImplementedError

    def reset_env(self, randomize_reset=False):
        self.env.reset(randomize=randomize_reset)
        obs = self.env.get_observation()
        return obs

    def reset_sim_env(self):
        obs, info = self.env.reset()
        return obs

    def process_sim_observation(self, raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
        assert (gripper_state <= 0.04).all(), gripper_state
        gripper_state = 1 - gripper_state / 0.04  # 1 is closed, 0 is open

        eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
        # conver quaternion to euler angles
        eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
        eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
        eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

        images = {}
        images["0"] = raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()
        images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()

        # wrist_img_resolution = (320, 240)
        # wrist_img = np.zeros(
        #     (len(images["2"]), wrist_img_resolution[1], wrist_img_resolution[0], 3)
        # )
        # for i in range(len(images["2"])):
        #     wrist_img[i] = cv2.resize(images["2"][i], wrist_img_resolution)
        # images["2"] = wrist_img

        obs = {
            "robot_state": {
                "joint_positions": joint_state,
                "gripper_position": gripper_state,
                "cartesian_position": eef_pos_euler,
            },
            "image": images,
        }
        return obs

    def unnormalize_action(self, naction):
        action = (naction + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min
        return action

    def unnormalized_delta_action(self, naction, state):
        action = (naction[:, :-1] + 1) * (
            self.delta_max - self.delta_min + 1e-6
        ) / 2 + self.delta_min
        action += state[:, -1, :-1]  # skip gripper
        gripper_action = (naction[:, -1:] + 1) * (
            self.action_max[-1] - self.action_min[-1] + 1e-6
        ) / 2 + self.action_min[-1]
        return np.concatenate([action, gripper_action], axis=-1)

    def unnormalized_sim_delta_action(self, naction, state):
        """
        naction: (batch_size, horizon, action_dim)
        state: (batch_size, cond_steps, obs_dim)
        """
        action = (naction[:, :, :-1] + 1) * (
            self.delta_max - self.delta_min + 1e-6
        ) / 2 + self.delta_min
        action += state[:, -1:, :-1]  # skip gripper
        gripper_action = (naction[:, :, -1:] + 1) * (
            self.action_max[-1] - self.action_min[-1] + 1e-6
        ) / 2 + self.action_min[-1]
        return np.concatenate([action, gripper_action], axis=-1)

    def unnormalize_obs(self, state):
        return (state + 1) * (self.obs_max - self.obs_min + 1e-6) / 2 + self.obs_min

    def postprocess_sim_gripper_action(self, action):
        action[..., -1] = -(action[..., -1] * 2 - 1)
        return action

    def process_multistep_state(self, obs, prev_obs=None):
        if self.n_cond_step == 2:
            assert prev_obs
            ret = np.stack(
                (
                    process_obs(
                        prev_obs, self.obs_min, self.obs_max, self.ordered_obs_keys
                    ),
                    process_obs(obs, self.obs_min, self.obs_max, self.ordered_obs_keys),
                ),
                axis=1,
            )
        else:
            assert self.n_cond_step == 1
            ret = process_obs(obs, self.obs_min, self.obs_max, self.ordered_obs_keys)[
                :, None
            ]
        ret = torch.from_numpy(ret).float().to(self.device)

        # TODO: use config
        # round gripper positionc
        if ret[..., -1] <= 0:
            ret[..., -1] = -1
        else:
            ret[..., -1] = 1
        # ret[..., -1] = torch.round(ret[..., -1])
        return ret

    def process_multistep_img(self, obs, camera_indices, prev_obs=None, bgr2rgb=False):
        if self.n_img_cond_step == 2:  # TODO: better logic
            assert prev_obs
            images = {}
            for idx in camera_indices:
                if len(obs["image"][idx].shape) == 3:
                    image_1 = obs["image"][idx][None].transpose(0, 3, 1, 2).copy()
                else:
                    image_1 = obs["image"][idx].transpose(0, 3, 1, 2).copy()
                if len(prev_obs["image"][idx].shape) == 3:
                    image_2 = prev_obs["image"][idx][None].transpose(0, 3, 1, 2).copy()
                else:
                    image_2 = prev_obs["image"][idx].transpose(0, 3, 1, 2).copy()
                images[idx] = np.stack(
                    (
                        image_1,
                        image_2,
                    ),
                    axis=1,
                )
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()
                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                # assert images[idx].shape == (1, 2, 3, 96, 96), images[idx].shape
        else:
            images = {}

            for idx in camera_indices:
                image = obs["image"][idx]
                if len(image.shape) == 3:
                    image = image[None]
                image = image.transpose(0, 3, 1, 2).copy()

                # Insert a cond_step dimension
                images[idx] = np.expand_dims(image, axis=1)
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()

                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                # assert images[idx].shape == (2, 1, 3, 192, 192), images[idx].shape
        return images
