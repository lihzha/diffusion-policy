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
from guided_dc.utils.pose_utils import quaternion_multiply, rotate_vectors
from guided_dc.utils.io_utils import load_json
from guided_dc.utils.traj_utils import interpolate_trajectory

DRAWER_COLLISION_BIT = 29

@register_env("Drawer-v1", max_episode_steps=100)
class Drawer(RandEnv):

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
        self._build_drawer()
        # sapien.set_log_level("warn")

    def _build_drawer(self):
        # we sample random drawer model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        train_json = '/home/irom-lab/projects/ManiSkill/mani_skill/assets/partnet_mobility/meta/info_cabinet_drawer_train.json'
        train_data = load_json(train_json)
        all_model_ids = np.array(list(train_data.keys()))
        rand_idx = self._episode_rng.permutation(np.arange(0, len(all_model_ids)))
        model_ids = all_model_ids[rand_idx]
        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(all_model_ids)).astype(int)
        )[: self.num_envs]
        link_ids = self._episode_rng.randint(0, 2**31, size=len(model_ids))

        self._drawers = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []
        drawer_links: List[List[Link]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            drawer_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            drawer_builder.set_scene_idxs(scene_idxs=[i])
            drawer = drawer_builder.build(name=f"{model_id}-{i}")

            # this disables self collisions by setting the group 2 bit at drawer_COLLISION_BIT all the same
            # that bit is also used to disable collision with the ground plane
            for link in drawer.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=DRAWER_COLLISION_BIT, bit=1
                )
            self._drawers.append(drawer)
            handle_links.append([])
            handle_links_meshes.append([])

            # TODO (stao): At the moment code for selecting semantic parts of articulations
            # is not very simple. Will be improved in the future as we add in features that
            # support part and mesh-wise annotations in a standard querable format
            for link, joint in zip(drawer.links, drawer.joints):
                if joint.type[0] in ["prismatic"]:
                    handle_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a handle
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, x: "handle" in x.name, mesh_name="handle"
                        )[0]
                    )

        # we can merge different articulations/links with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.drawer = Articulation.merge(self._drawers, name="drawer")
        
        # self.handle_link = Link.merge(
        #     [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
        #     name="handle_link",
        # )
        
        self.handle_link = Link.merge(
            [links[-1] for i, links in enumerate(handle_links)],
            name="handle_link",
        )

        # store the position of the handle mesh itself relative to the link it is apart of
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            )
        )
        

        self.handle_link_normal_raw = common.to_tensor(
            np.array(
                [
                    [-1.,0.,0.] for i, meshes in enumerate(handle_links_meshes)
                ]
            )
        )

        self.handle_link_goal = actors.build_box(
            self.scene,
            half_sizes=[0.05, 0.01, 0.01],
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
        )
        
    def _after_reconfigure(self, options: dict):

        # this value is used to set object pose so the bottom is at z=0

        self.drawer_length = []
        self.drawer_width = []
        self.drawer_height = []

        for drawer in self._drawers:
            bbox = drawer.get_first_collision_mesh().bounding_box.bounds
            self.drawer_length.append(np.maximum(np.abs(bbox[0, 0]-bbox[1,0]), np.abs(bbox[0, 1]-bbox[1,1])))
            self.drawer_width.append(np.minimum(np.abs(bbox[0, 0]-bbox[1,0]), np.abs(bbox[0, 1]-bbox[1,1])))
            self.drawer_height.append(np.abs(bbox[0, 2]-bbox[1,2])/2)
        
        print(self.drawer_height)
        self.drawer_height = common.to_tensor(self.drawer_height)
        self.drawer_width = common.to_tensor(self.drawer_width)
        self.drawer_length = common.to_tensor(self.drawer_length)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: torch.Tensor = None):
        if env_idx is None:
            return transform_points(
                    self.handle_link.pose.to_transformation_matrix().clone(),
                    common.to_tensor(self.handle_link_pos),
                )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx]),
        )
    
    def handle_link_orientations(self, env_idx: torch.Tensor = None):

        approaching = -self.handle_link_normal  # already rotated in the global frame
        closing = np.array([0,0,1])   # global frame
        center = self.handle_link_positions()  # global frame
        sapien_poses = [self.agent.build_grasp_pose(approaching[i].cpu().numpy(), closing, center[i].cpu().numpy()) for i in range(len(approaching))]
        
        p = common.to_tensor(np.stack([pose.p for pose in sapien_poses], axis=0))
        q = common.to_tensor(np.stack([pose.q for pose in sapien_poses], axis=0))
        # handle_link_quat = torch.cat((p,q), dim=1)   # global

        if env_idx is None:
            return q
        else:
            return q[env_idx]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
        
            robot_pos: torch.Tensor = self._randomize_robot_pos()
            self.table_scene.initialize(env_idx, robot_pos)
            lighting = self._randomize_lighting()
            camera_pose = self._randomize_camera_pose()
            self._randomize_obj_pos(env_idx)
            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            extra_state_dict = dict(lighting=lighting, camera=camera_pose)
            self.save_start_state(extra_state_dict)
            
            self.handle_link_goal.set_pose(
                    Pose.create_from_pq(p=self.handle_link_positions(env_idx), q=self.handle_link_orientations(env_idx))
                )
            # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            qlimits = self.drawer.get_qlimits()  # [b, self.cabinet.max_dof, 2])
            self.drawer.set_qpos(qlimits[env_idx, :, 0])
            self.drawer.set_qvel(self.drawer.qpos[env_idx] * 0)
            # NOTE (stao): This is a temporary work around for the issue where the cabinet drawers/doors might open
            # themselves on the first step. It's unclear why this happens on GPU sim only atm.
            # moreover despite setting qpos/qvel to 0, the cabinets might still move on their own a little bit.
            # this may be due to oblong meshes.
            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            
            print('Initialized episode.')
        
    def evaluate(self):
        # even though self.handle_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()
        # TODO (stao): setting the pose of the visual sphere here seems to cause mayhem with cabinet qpos
        # self.handle_link_goal.set_pose(Pose.create_from_pq(p=self.handle_link_positions()))
        # self.scene._gpu_apply_all()
        # self.scene._gpu_fetch_all()
        # update the goal sphere to its new position
        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose.clone(),
            gripper=self.agent.robot.get_qpos()[:, -1].clone()
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reaching_reward[
            amount_to_open_left < 0.999
        ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
        # print(open_reward.shape)
        open_reward[info["open_enough"]] = 3  # give max reward here
        reward = reaching_reward + open_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def _randomize_obj_pos(self, env_idx):
        # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
        b = len(env_idx)
    
        drawer_xyz = torch.zeros((b,3))
        drawer_xyz[:, 2] = self.drawer_height

        # Randomize place obj position
        drawer_xyz[:, 0] = randomize_by_percentage(
            self.drawer_length, 
            low_percentage=self.rand_cfg.goal.low_percentage_x,
            high_percentage=self.rand_cfg.goal.high_percentage_x
            )
        drawer_xyz[:, 1] = randomize_by_percentage(
            self.drawer_width, 
            low_percentage=self.rand_cfg.goal.low_percentage_y,
            high_percentage=self.rand_cfg.goal.high_percentage_y,
            )

        # Randomize place obj rotation
        drawer_quat = torch.zeros((b,4))
        q = [1, 0, 0, 0]
        q_delta_zrot_max = self.rand_cfg.object.manip_obj_delta_zrot_max
        q_delta_zrot_min = self.rand_cfg.object.manip_obj_delta_zrot_min
        q_delta_zrot = randomize_continuous(q_delta_zrot_min, q_delta_zrot_max, b, return_list=True)
        for i, zrot in enumerate(q_delta_zrot):
            drawer_quat[i] = common.to_tensor(quaternion_multiply(q, euler2quat(0, 0, zrot)))
        
        self.drawer.set_pose(Pose.create_from_pq(p=drawer_xyz, q=drawer_quat))
        # if isinstance(q_delta_zrot, float):
        #     q_delta_zrot = np.array([q_delta_zrot])
        self.handle_link_normal = torch.zeros_like(self.handle_link_normal_raw)
        for i, zrot in enumerate(q_delta_zrot):
            self.handle_link_normal[i] = rotate_vectors(self.handle_link_normal_raw[i].unsqueeze(0), 
                                                        common.to_tensor(zrot) if len(q_delta_zrot)>1 else torch.tensor(zrot, device=self.device))
         
    def get_grasp_pose(self):
        pose = Pose.create_from_pq(p=self.handle_link_positions(), q=self.handle_link_orientations().to(torch.float))
        return pose
    
    def get_target_qpos(self):
        
        goal_cfg_at_global: sapien.Pose = copy.deepcopy(self.get_grasp_pose()) # in global frame
        q0 = self.agent.robot.get_qpos().clone()
        goal_cfg_at_base = self.agent.robot.pose.inv() * goal_cfg_at_global
        goal_qpos_for_arm = self.agent.controller.controllers['arm'].kinematics.compute_ik(target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False)
        goal_qpos_for_gripper = torch.ones_like(goal_qpos_for_arm)[:, :2]  # 2 dim, one gripper hand mimic the other so just set them the same
        qpos_to_set = torch.cat((goal_qpos_for_arm, goal_qpos_for_gripper), dim=-1)
        
        pull_direction: torch.Tensor = self.unwrapped.handle_link_normal.clone()
        goal_pose_for_pull = Pose.create_from_pq(p=goal_cfg_at_global.p+pull_direction*0.2, q=goal_cfg_at_global.q)
        goal_pose_for_pull_at_base = self.agent.robot.pose.inv() * goal_pose_for_pull
        goal_qpos_for_pull_arm = self.agent.controller.controllers['arm'].kinematics.compute_ik(target_pose=goal_pose_for_pull_at_base, q0=q0, use_delta_ik_solver=False)
        goal_qpos_for_pull_gripper = torch.ones_like(goal_qpos_for_pull_arm)[:, :1]*(-1)
        qpos_to_set_pull_start_arm = self.agent.controller.controllers['arm'].kinematics.compute_ik(target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False)
        qpos_to_set_pull_start = torch.cat((qpos_to_set_pull_start_arm, goal_qpos_for_pull_gripper), dim=-1)
        qpos_to_set_for_pull = torch.cat((goal_qpos_for_pull_arm, goal_qpos_for_pull_gripper), dim=-1)
        qpos_paths_for_pull = interpolate_trajectory(torch.cat((qpos_to_set_pull_start.unsqueeze(1), qpos_to_set_for_pull.unsqueeze(1)), dim=1), 10)
    
        return qpos_to_set, qpos_paths_for_pull
    
    def visualize_mesh(self, meshes):

        import matplotlib.pyplot as plt

        for i, handle_mesh in enumerate(meshes):
            # Sample points from the mesh surface (point cloud)
            num_points = 10000
            point_cloud = handle_mesh.sample(num_points)
            face_centroids = self.handle_link_pos[i].cpu().numpy()
            face_normals = self.handle_link_normal[i].cpu().numpy()
            # face_centroids = handle_mesh.bounding_box.center_mass
            # face_normals = get_normal_axis_direction(handle_mesh).squeeze()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the point cloud of the mesh
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='cyan', s=2, label='Point Cloud')

            # Plot the face normals (centroids + normal vectors)
            ax.quiver(face_centroids[0], face_centroids[1], face_centroids[2],
                    face_normals[0], face_normals[1], face_normals[2], length=0.05, color='red', label='Face Normals')

            # Labels and plot settings
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # Save the plot as an image file (e.g., PNG)
            # plt.savefig('mesh_normals_with_grasp_pose.png', dpi=300)
            plt.show()

            # Optional: close the plot if you're running this in a script
            plt.close()