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

from typing import Any, ClassVar, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array

from guided_dc.envs.randomized_env import RandEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder


@register_env("PushObject-v1", max_episode_steps=50)
class PushObjectEnv(RandEnv):
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

    SUPPORTED_ROBOTS: ClassVar[list] = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.init_cfg = kwargs.pop("init")
        self.robot_init_qpos_noise = self.init_cfg.robot.robot_init_qpos_noise
        super().__init__(
            *args, robot_uids=robot_uids, rand_cfg=kwargs.pop("rand"), **kwargs
        )

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TabletopSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )

        self.cam_mount = self.scene.create_actor_builder().build_kinematic(
            "camera_mount"
        )
        self.table_scene.build()

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self._build_manip_obj(self.init_cfg.object.manip_obj)

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_obj = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_obj",
            add_collision=False,
            body_type="kinematic",
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in pushCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        self._hidden_objects.append(self.goal_obj)

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

    def _build_manip_obj(self, obj_name):
        if obj_name == "cube":
            self.manip_obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="cube",
                body_type="dynamic",
            )

        else:
            self.manip_obj, self._manip_objs = self._load_assets(obj_name)

    def _after_reconfigure(self, options: dict):
        self.manip_obj_height = []
        for obj in self._manip_objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.manip_obj_height.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.manip_obj_height = common.to_tensor(self.manip_obj_height)
        self.goal_obj_length = common.to_tensor([self.goal_radius] * self.num_envs)
        self.goal_obj_width = common.to_tensor([self.goal_radius] * self.num_envs)
        self.goal_obj_height = common.to_tensor([1e-2] * self.num_envs)

    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        is_obj_placed = (
            torch.linalg.norm(
                self.manip_obj.pose.p[..., :2] - self.goal_obj.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_obj.pose.p,
                obj_pose=self.manip_obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.manip_obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.manip_obj.pose.p[..., :2] - self.goal_obj.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
