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

from typing import Any, Dict, Union, List
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat
import sapien.physx as physx
import os
import copy
import trimesh
import logging

from mani_skill.utils.structs.types import Array
from mani_skill.agents.robots import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.structs import Articulation, Link, Pose, Actor
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.geometry import transform_points

from guided_dc.envs.wrappers.randomized_env import RandEnv
from guided_dc.envs.scenes.tabletop_scene_builder import TabletopSceneBuilder
from guided_dc.utils.randomization import randomize_continuous, randomize_by_percentage
from guided_dc.utils.pose_utils import quaternion_multiply, rotate_vectors, normal_to_forward_quat
from guided_dc.utils.io_utils import load_json
from guided_dc.utils.traj_utils import interpolate_trajectory
from guided_dc.utils.pointcloud_utils import PointCloud
from guided_dc.utils.vis_utils import visualize_trimesh

TOP_DOWN_GRASP_BIAS: float = 1e-4
NUM_SAMPLES = 20

@register_env("PickAndPlace-v1", max_episode_steps=100)
class PickAndPlace(RandEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    agent: Union[Panda]

    min_open_frac = 0.75
  
    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.init_cfg = kwargs['init']
        if isinstance(self.init_cfg, dict):
            from omegaconf import OmegaConf
            self.init_cfg = OmegaConf.create(self.init_cfg)
        self.robot_init_qpos_noise = self.init_cfg.robot.robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, rand_cfg=kwargs.pop('rand'), init_cfg=kwargs.pop('init'), **kwargs)

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        
        floor_texture_file_path = '/home/irom-lab/projects/guided-data-collection/guided_dc/assets/floor'
        table_model_file_path = '/home/irom-lab/projects/guided-data-collection/guided_dc/assets/table'
        # Get all files in table_model_file_path that ends with .obj
        all_table_model_files = [f for f in os.listdir(table_model_file_path) if f.endswith('.obj')]
        # Get all files in floor_texture_file_path that ends with .png or .jpg or .jpeg
        all_floor_texture_files = [f for f in os.listdir(floor_texture_file_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        
        # Randomly choose a table model file and a floor texture file from the list of files
        table_model_file = os.path.join(table_model_file_path, np.random.choice(all_table_model_files))
        floor_texture_file = os.path.join(floor_texture_file_path, np.random.choice(all_floor_texture_files))
        
        self.table_scene = TabletopSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise, table_model_file=table_model_file, floor_texture_file=floor_texture_file
        )

        self.cam_mount = [self.scene.create_actor_builder().build_kinematic(f"camera_mount_{i}") for i in range(len(self.init_cfg.camera.eye))]
        
        self.table_scene.build()
        sapien.set_log_level("off")
        self._build_pick_and_place_obj()
        # sapien.set_log_level("warn")

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
        

        # self.handle_link_normal_raw = common.to_tensor(
        #     np.array(
        #         [
        #             [-1.,0.,0.] for i, meshes in enumerate(handle_links_meshes)
        #         ]
        #     )
        # )

        self.grasp_point = actors.build_box(
            self.scene,
            half_sizes=[0.05, 0.01, 0.01],
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
        )
        
    def _after_reconfigure(self, options: dict):

        self.manip_obj_meshes_raw = []
        self.manip_obj_height = []
        for obj in self._manip_objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.manip_obj_meshes_raw.append(collision_mesh)
           
            # this value is used to set object pose so the bottom is at z=0
            self.manip_obj_height.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.manip_obj_height = common.to_tensor(self.manip_obj_height)

        self.goal_obj_height = []
        self.goal_obj_length = []
        self.goal_obj_width = []
        self.goal_obj_meshes_raw = []
        for obj in self._goal_objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.goal_obj_meshes_raw.append(collision_mesh)
            bbox = collision_mesh.bounding_box.bounds
            
            self.goal_obj_height.append(-bbox[0, 2])
            self.goal_obj_length.append(np.maximum(np.abs(bbox[0, 0])*2, np.abs(bbox[0, 1])*2))
            self.goal_obj_width.append(np.minimum(np.abs(bbox[0, 0])*2, np.abs(bbox[0, 1])*2))
        self.goal_obj_height = common.to_tensor(self.goal_obj_height)
        self.goal_obj_width = common.to_tensor(self.goal_obj_width)
        self.goal_obj_length = common.to_tensor(self.goal_obj_length)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
        
            robot_pos: torch.Tensor = self._randomize_robot_pos()
            self.table_scene.initialize(env_idx, robot_pos)
            lighting = self._randomize_lighting()
            camera_pose = self._randomize_camera_pose()
            self._randomize_obj_pos(env_idx, self.manip_obj, self.goal_obj)
            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            extra_state_dict = dict(lighting=lighting, camera=camera_pose)
            self.save_start_state(extra_state_dict)
            
            self.manip_obj_pcds = []
            self.goal_obj_pcds = []
            if len(self.manip_obj_meshes_raw) > 1:
                for i in range(len(self.manip_obj_meshes_raw)):
                    manip_mesh = self.manip_obj_meshes_raw[i].copy()
                    goal_mesh = self.goal_obj_meshes_raw[i].copy()
                    manip_pose = self.manip_obj.pose[i]
                    goal_pose = self.goal_obj.pose[i]
                    
                    manip_mesh.apply_transform(manip_pose.sp.to_transformation_matrix())
                    goal_mesh.apply_transform(goal_pose.sp.to_transformation_matrix())
                
                    samples = trimesh.sample.sample_surface_even(manip_mesh, NUM_SAMPLES)
                    pcd = PointCloud(xyz_pts=samples[0].__array__().copy(), normals=manip_mesh.face_normals[samples[1]].copy())
                    self.manip_obj_pcds.append(pcd)

                    # visualize_trimesh(manip_mesh, extra_pts=(samples[0].__array__().copy(), manip_mesh.face_normals[samples[1]].copy()))
                    
                    samples = trimesh.sample.sample_surface_even(goal_mesh, NUM_SAMPLES)
                    pcd = PointCloud(xyz_pts=samples[0].__array__().copy(), normals=goal_mesh.face_normals[samples[1]].copy())
                    self.goal_obj_pcds.append(pcd)
            else:
                manip_mesh = self.manip_obj_meshes_raw.copy()
                goal_mesh = self.goal_obj_meshes_raw.copy()
                manip_pose = self.manip_obj.pose
                goal_pose = self.goal_obj.pose
                
                manip_mesh.apply_transform(manip_pose.sp.to_transformation_matrix())
                goal_mesh.apply_transform(goal_pose.sp.to_transformation_matrix())
            
                samples = trimesh.sample.sample_surface_even(manip_mesh, NUM_SAMPLES)
                pcd = PointCloud(xyz_pts=samples[0].__array__().copy(), normals=manip_mesh.face_normals[samples[1]].copy())
                self.manip_obj_pcds.append(pcd)
                
                samples = trimesh.sample.sample_surface_even(goal_mesh, NUM_SAMPLES)
                pcd = PointCloud(xyz_pts=samples[0].__array__().copy(), normals=goal_mesh.face_normals[samples[1]].copy())
                self.goal_obj_pcds.append(pcd)
            
            
            print('Initialized episode.')
        
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
            tcp_pose=self.agent.tcp.pose.raw_pose.clone(),
            gripper=self.agent.robot.get_qpos()[:, -1].clone(),
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
    
    def get_grasp_pose(self):
        pose = self.get_feasible_candidate(self.manip_obj_pcds)
        self.grasp_point.set_pose(pose)
        if physx.is_gpu_enabled():
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()
        return pose
    
    def get_target_qpos(self):
        goal_cfg_at_global: sapien.Pose = copy.deepcopy(self.get_grasp_pose())
        q0 = self.agent.robot.get_qpos().clone()
        goal_cfg_at_base = self.agent.robot.pose.inv() * goal_cfg_at_global
        goal_qpos_for_arm = self.agent.controller.controllers['arm'].kinematics.compute_ik(target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False)
        goal_qpos_for_gripper = torch.ones_like(goal_qpos_for_arm)[:, :2]  # 2 dim, one gripper hand mimic the other so just set them the same
        qpos_to_set = torch.cat((goal_qpos_for_arm, goal_qpos_for_gripper), dim=-1)
        
        return qpos_to_set, None

    def get_feasible_candidate(self, link_point_clouds: List[PointCloud], pushin_more: bool = True):
        
        grasp_p = np.zeros((len(link_point_clouds), 3))
        grasp_q = np.zeros((len(link_point_clouds), 4))
        
        for pcd_idx, link_point_cloud in enumerate(link_point_clouds):
            normals = link_point_cloud.normals
            pointing_down = normals[:, 2] < 0.0

            p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
            if not pointing_down.all():
                # sample the ones point up more often
                p = np.exp(normals[:, 2] * TOP_DOWN_GRASP_BIAS)
            p /= p.sum()

            candidate = None
            errors = []

            candidate_indices = self._episode_rng.choice(
                len(link_point_cloud),
                size=min(self.init_cfg.grasp.num_action_candidates, len(link_point_cloud)),
                p=p,
                replace=True,
            )
            for i, idx in enumerate(candidate_indices):
                assert i<= 1
                position = link_point_cloud.xyz_pts[idx].copy()
                grasp_to_ee = self.agent.tcp.pose.p.cpu().numpy()[pcd_idx] - position  # in global frame, no broadcasting
                grasp_to_ee /= np.linalg.norm(grasp_to_ee)  # if broadcasting, axis=1
                if pointing_down.all():
                    # disallow bottom up grasps, so using point to ee
                    # as normal, with some random exploration
                    grasp_to_ee += self._episode_rng.randn(3) * 0.1 # if broadcasting, randn is 2D
                    grasp_to_ee /= np.linalg.norm(grasp_to_ee)  # if broadcasting, axis=1
                    normal = grasp_to_ee
                else:
                    normal = normals[idx]
                    # # Expand normal to (len(grasp_to_ee), 3)
                    # normal = np.tile(normal, (len(grasp_to_ee), 1))
                    
                ## Orient normal towards end effector
                ## For broadcasting, get dot product of each row of grasp_to_ee and normal
                # dot_product = np.einsum('ij,ij->i', grasp_to_ee, normal) 
                # neg_indices = dot_product < 0
                # normal[neg_indices] *= -1.0
                if np.dot(grasp_to_ee, normal) < 0:
                    normal *= -1.0
                # compute base orientation, randomize along Z-axis
                try:
                    # base_orientation = quaternion_multiply(  # If broadcasting, using batch_quaternion_multiply
                    #     normal_to_forward_quat(normal),
                    #     euler2quat(-np.pi, 0, np.pi),
                    # )
                    base_orientation = normal_to_forward_quat(normal)
                except np.linalg.LinAlgError as e:
                    logging.warning(e)
                    base_orientation = euler2quat(0, 0, 0)
                # z_angle = self._episode_rng.uniform(np.pi, -np.pi)
                # z_orn = euler2quat(0, 0, z_angle)
                # base_orientation = quaternion_multiply(base_orientation, z_orn)
                # pregrasp_distance = self._episode_rng.uniform(0.1, 0.2)
                if pushin_more:
                    # prioritize grasps that push in more, if possible, but slowly
                    # back off to prevent all grasps colliding.
                    pushin_distance = (len(candidate_indices) - i) / len(
                        candidate_indices
                    ) * (
                        self.init_cfg.grasp.max_pushin_dist - self.init_cfg.grasp.min_pushin_dist
                    ) + self.init_cfg.grasp.min_pushin_dist
                else:
                    # prioritize grasps that push in less. Useful for "pushing"
                    pushin_distance = (
                        i
                        / len(candidate_indices)
                        * (self.init_cfg.grasp.max_pushin_dist - self.init_cfg.grasp.min_pushin_dist)
                        + self.init_cfg.grasp.min_pushin_dist
                    )

                grasp_p[pcd_idx] = position - normal * pushin_distance
                grasp_q[pcd_idx] = base_orientation

        # compute grasp pose relative to object
        grasp_pose = Pose.create_from_pq(
            p=grasp_p,
            q=grasp_q,
        )
        
        return grasp_pose
    #     candidate = GraspLinkPoseAction(
    #         link_path=self.link_path,
    #         pose=grasp_pose,
    #         with_backup=self.with_backup,
    #         backup_distance=pushin_distance + pregrasp_distance,
    #     )
    #     try:
    #         candidate.check_feasibility(env=env)
    #         logging.info(
    #             f"[{i}|{env.episode_id}] "
    #             + f"pushin_distance: {pushin_distance} ({self.pushin_more})"
    #         )
    #         break
    #     except Action.InfeasibleAction as e:
    #         errors.append(e)
    #         candidate = None
    #         continue
    # del normals, link_point_cloud, grasp_to_ee, candidate_indices, p

    # # see `tests/test_memory_leak.py``, this needs to be here,
    # # otherwise, a few MBs will leak each time
    # if candidate is not None:
    #     return ActionListPolicy(actions=[candidate])
    # raise Action.InfeasibleAction(
    #     action_class_name=type(self).__name__,
    #     message=f"all candidates failed {self}:"
    #     + ",".join(list({str(e) for e in errors})[:3]),
    # )