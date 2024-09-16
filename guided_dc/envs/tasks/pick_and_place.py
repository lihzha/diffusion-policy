"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import Array
from mani_skill.utils.structs import Pose
from mani_skill.utils import common

from guided_dc.envs.wrappers.randomized_env import RandEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder

@register_env("PickAndPlace-v1", max_episode_steps=50)
class PickAndPlaceEnv(RandEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to push and move a cube to a goal region in front of it

    Randomizations
    --------------
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    Success Conditions
    ------------------
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.

    Visualization: https://maniskill.readthedocs.io/en/latest/tasks/index.html#pushcube-v1
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]
  
    def __init__(self, *args, robot_uids="panda", **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.init_cfg = kwargs.pop('init')
        self.robot_init_qpos_noise = self.init_cfg.robot.robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, rand_cfg=kwargs.pop('rand'), **kwargs)

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TabletopSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )

        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        self.table_scene.build()

        self._build_pick_and_place_obj()

    def _after_reconfigure(self, options: dict):

        self.manip_obj_height = []
        for obj in self._manip_objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.manip_obj_height.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.manip_obj_height = common.to_tensor(self.manip_obj_height)

        self.goal_obj_height = []
        self.goal_obj_length = []
        self.goal_obj_width = []
        for obj in self._goal_objs:
            bbox = obj.get_first_collision_mesh().bounding_box.bounds
            self.goal_obj_height.append(-bbox[0, 2])
            self.goal_obj_length.append(np.maximum(np.abs(bbox[0, 0])*2, np.abs(bbox[0, 1])*2))
            self.goal_obj_width.append(np.minimum(np.abs(bbox[0, 0])*2, np.abs(bbox[0, 1])*2))
        self.goal_obj_height = common.to_tensor(self.goal_obj_height)
        self.goal_obj_width = common.to_tensor(self.goal_obj_width)
        self.goal_obj_length = common.to_tensor(self.goal_obj_length)

    def _build_pick_and_place_obj(self):

        material = {
            'static_friction':self.init_cfg.physics.static_friction,
            'dynamic_friction':self.init_cfg.physics.dynamic_friction,
            'restitution':self.init_cfg.physics.restitution
        }

        physical_material = sapien.Scene().create_physical_material(**material)

        # If num_envs > 1, there will be multiple objects.
        self.manip_obj, self._manip_objs = self._load_assets(self.init_cfg.object.manip_obj, physical_material)
        self.goal_obj, self._goal_objs = self._load_assets(self.init_cfg.object.goal_obj, physical_material)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            robot_pos: sapien.Pose = self._randomize_robot_pos()
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx, robot_pos)

            self._randomize_lighting()
            self._randomize_camera_pose()
            self._randomize_obj_pos(env_idx, self.manip_obj, self.goal_obj)
            
    def evaluate(self):
        pos_pick = self.manip_obj.pose.p
        pos_place = self.goal_obj.pose.p
        offset = pos_pick - pos_place
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.manip_obj_height/2 - self.goal_obj_height/2) <= 0.005  # TODO: replace block_half_size
        )
        is_obj_on_place = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.manip_obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.manip_obj)
        success = is_obj_on_place & is_obj_static & (~is_obj_grasped)
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_place": is_obj_on_place,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.goal_obj.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.manip_obj.pose.raw_pose,
                tcp_to_obj_pos=self.manip_obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.manip_obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward TODO: need object specific metrics  
        obj_pos = self.manip_obj.pose.p
        place_obj_top_pos = self.goal_obj.pose.p.clone()
        place_obj_top_pos[:, 2] = place_obj_top_pos[:, 2] + self.manip_obj_height/2 + self.manip_obj_height/2
        obj_to_place_obj_top_dist = torch.linalg.norm(place_obj_top_pos - obj_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * obj_to_place_obj_top_dist)
        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_obj_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.manip_obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.manip_obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(
            0.2
        )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_place"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_place"]]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
