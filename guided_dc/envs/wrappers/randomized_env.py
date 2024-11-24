import torch
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from pathlib import Path
import sapien.physx as physx
import logging

from mani_skill.envs.utils import randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor

from guided_dc.utils.randomization import randomize_continuous, randomize_by_percentage
from guided_dc.utils.pose_utils import quaternion_multiply, normal_to_forward_quat
from guided_dc.utils.obj_utils import get_obj_asset


from typing import List

FORCE_THRESHOLD = 1e-5

class RandEnv(BaseEnv):
    def __init__(self, *args, robot_uids="panda", rand_cfg=None, init_cfg=None, num_envs=1, reconfiguration_freq=None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.rand_cfg = rand_cfg
        if isinstance(self.rand_cfg, dict):
            from omegaconf import OmegaConf
            self.rand_cfg = OmegaConf.create(self.rand_cfg)
            self.init_cfg = OmegaConf.create(init_cfg)
        else:
            self.init_cfg = init_cfg
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 0
            else:
                reconfiguration_freq = 0
        self.start_state = None
        super().__init__(*args, robot_uids=robot_uids, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        assert isinstance(self.cam_mount, list)
        camera_configs = []
        for i, cam_mount in enumerate(self.cam_mount):
            camera_configs.append(
                CameraConfig(
                    f"camera_{i}",
                    pose=sapien.Pose(),
                    width=512,
                    height=512,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100, 
                    mount=cam_mount)
                )
        return camera_configs + self.agent._sensor_configs

    @property
    def _default_viewer_camera_configs(
        self,
    ) -> CameraConfig:
        """Default configuration for the viewer camera, controlling shader, fov, etc. By default if there is a human render camera called "render_camera" then the viewer will use that camera's pose."""
        return CameraConfig(uid="viewer", pose=sapien.Pose([0, 0, 1]), width=1920, height=1080, shader_pack="default", near=0.0, far=1000, mount=self.cam_mount[0])


    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=sapien.Pose(), width=1024, height=1024, fov=1, near=0.01, far=100, mount=self.cam_mount[0]
        )


    def _randomize_lighting(self, lighting=None):
        
        shadow = self.enable_shadow
        if lighting is None:
            ambient_min = self.rand_cfg.lighting.ambient_min
            ambient_max = self.rand_cfg.lighting.ambient_max
            ambient_light = randomize_continuous(min=ambient_min, max=ambient_max, size=3)
        else:
            ambient_light = lighting
        self.scene.set_ambient_light(ambient_light)
        return ambient_light
        #TODO: how to remove previous directional lights
        # self.scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        # )
        # self.scene.add_directional_light([0, 0, -1], [1, 1, 1])
    
    def _randomize_camera_pose(self, camera_pose=None):
        rad_range = self.rand_cfg.camera.rad_range
        if len(self.cam_mount) == 1:
            if camera_pose is None:
                pose = sapien_utils.look_at(eye=self.init_cfg.camera.eye, target=self.init_cfg.camera.target)
                pose = Pose.create(pose)
                pose = pose * Pose.create_from_pq(
                    p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                    q=randomization.random_quaternions(
                        n=self.num_envs, device=self.device, bounds=(-rad_range, rad_range)
                    ),
                )
            else:
                pose = Pose.create(camera_pose)
            self.cam_mount.set_pose(pose)
            return [pose.raw_pose.cpu()]
        else:
            poses = []
            for i, cam_mount in enumerate(self.cam_mount):
                if camera_pose is None:
                    pose = sapien_utils.look_at(eye=self.init_cfg.camera.eye[i], target=self.init_cfg.camera.target[i])
                    pose = Pose.create(pose)
                    pose = pose * Pose.create_from_pq(
                        p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                        q=randomization.random_quaternions(
                            n=self.num_envs, device=self.device, bounds=(-rad_range, rad_range)
                        ),
                    )
                else:
                    pose = Pose.create(camera_pose[i])
                cam_mount.set_pose(pose)
                poses.append(pose.raw_pose.cpu())
            return torch.stack(poses, dim=0)

    def _randomize_robot_pos(self):
        x_min = self.rand_cfg.robot.pos_range.x_min
        x_max = self.rand_cfg.robot.pos_range.x_max
        y_min = self.rand_cfg.robot.pos_range.y_min
        y_max = self.rand_cfg.robot.pos_range.y_max
        x_pos = randomize_continuous(x_min, x_max, 1)
        y_pos = randomize_continuous(y_min, y_max, 1)
        xyz_pos = [x_pos, y_pos, 0]
        return np.array(xyz_pos)

    def _randomize_obj_pos(self, env_idx, manip_obj, goal_obj=None):
        # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
        b = len(env_idx)

        # Randomize manip obj position
        pos_scale = self.rand_cfg.object.pos_scale
        pos_offset = self.rand_cfg.object.pos_offset
        xyz = torch.zeros((b, 3))
        xyz[..., :2] = torch.rand((b, 2)) * pos_scale + pos_offset
        xyz[..., 2] = self.manip_obj_height * 1.1

        # Randomize manip obj rotation
        q = [1, 0, 0, 0]
        manip_obj_q = torch.zeros((b, 4))
        q_delta_zrot_max = self.rand_cfg.object.manip_obj_delta_zrot_max
        q_delta_zrot_min = self.rand_cfg.object.manip_obj_delta_zrot_min
        q_delta_zrot = randomize_continuous(q_delta_zrot_min, q_delta_zrot_max, b)
        for i, zrot in enumerate(q_delta_zrot):
            manip_obj_q[i] = common.to_tensor(quaternion_multiply(q, euler2quat(0, 0, zrot)))
        obj_pose = Pose.create_from_pq(p=xyz, q=manip_obj_q) # Pose.create_from_pq will automatically batch p or q accordingly
        manip_obj.set_pose(obj_pose)

        if goal_obj:
            # Randomize goal obj position
            goal_obj_x = randomize_by_percentage(
                self.goal_obj_length, 
                low_percentage=self.rand_cfg.goal.low_percentage_x,
                high_percentage=self.rand_cfg.goal.high_percentage_x
            )
            goal_obj_y = randomize_by_percentage(
                self.goal_obj_width, 
                low_percentage=self.rand_cfg.goal.low_percentage_y,
                high_percentage=self.rand_cfg.goal.high_percentage_y
            )
            goal_obj_xyz = torch.stack((goal_obj_x, goal_obj_y, self.goal_obj_height), dim=1)
            # Randomize goal obj rotation
            q_delta_zrot_max = self.rand_cfg.object.goal_obj_delta_zrot_max
            q_delta_zrot_min = self.rand_cfg.object.goal_obj_delta_zrot_min
            q_delta_zrot = randomize_continuous(q_delta_zrot_min, q_delta_zrot_max, b)
            goal_obj_q = torch.zeros((b, 4))
            for i, zrot in enumerate(q_delta_zrot):
                goal_obj_q[i] = common.to_tensor(quaternion_multiply(q, euler2quat(self.init_cfg.goal.raw, self.init_cfg.goal.yaw, zrot)))
            goal_obj.set_pose(Pose.create_from_pq(p=goal_obj_xyz.to(torch.float), q=goal_obj_q.to(torch.float)))

    # Overload the _load_lighting() functions in sapien_env
    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])
    
    def _load_assets(self, obj_name, physical_material=None):
        asset_path_list: list = get_obj_asset(obj_name)
        rand_idx = torch.randperm(len(asset_path_list))
        model_files = [asset_path_list[idx] for idx in rand_idx]
        model_files = np.concatenate(
            [model_files] * np.ceil(self.num_envs / len(asset_path_list)).astype(int)
        )[: self.num_envs]
        if (
            self.num_envs > 1
            and self.num_envs < len(asset_path_list)
            and self.reconfiguration_freq <= 0
        ):
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be > 1."""
            )

        _objs: List[Actor] = []
        for i, model_file in enumerate(model_files):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            model_name = Path(model_file).name
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])

            scale = 1
            obj_pose = sapien.Pose(q=euler2quat(0, 0, 0))

            if physical_material is not None:
                builder.add_convex_collision_from_file(
                    filename=model_file,
                    scale=[scale] * 3,
                    pose=obj_pose,
                    material=physical_material
                )
            else:
                print(f"Using default physical properties for the object {model_name}.")
                builder.add_convex_collision_from_file(
                    filename=model_file,
                    scale=[scale] * 3,
                    pose=obj_pose
                )

            builder.add_visual_from_file(
                filename=model_file, scale=[scale] * 3, pose=obj_pose
            )
            _objs.append(builder.build(name=f"{model_name}-{i}"))

        obj = Actor.merge(_objs, name="custom_assets")

        return obj, _objs

    # def collision_checker(self, new_cfg):
    #     init_state = self.scene.get_sim_state()
    #     # TODO: make sure new_cfg is qpos
    #     self.agent.robot.set_qpos(new_cfg)
    #     self.step(np.zeros_like(self.action_space.sample()))

    #     # 1. Get all objects' links
    #     link_lists = [drawer.get_links() for drawer in self._drawers]
    #     link_num = []
    #     all_object_links = []
    #     for link_list in link_lists:
    #         link_num.append(len(link_list))
    #         all_object_links += link_list
    #     assert len(set(link_num)) == 1, print('Drawers in different envs have different numbers of links!')
        
    #     # TODO: table
    #     all_actors = [self.table_scene.table]
        
    #     # 2. Get robots' links
    #     robot_links = self.agent.robot.links   # list
    #     has_collision = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)
    #     for robot_link in robot_links:
    #         for link in all_object_links:
    #             force = self.scene.get_pairwise_contact_forces(robot_link, link)
    #             if force.shape[0] == 1:
    #                 if torch.norm(force) < FORCE_THRESHOLD:
    #                     has_collision[link._scene_idxs] = True
    #             elif force.shape[0] == self.num_envs:
    #                 has_collision = has_collision | (torch.norm(force, dim=1) < FORCE_THRESHOLD)
    #             else:
    #                 raise ValueError('The contact forces shape does not match num_envs.')
    #         # TODO: figure out the order of env
    #         for actor in all_actors:
    #             force = self.scene.get_pairwise_contact_forces(robot_link, actor)
    #             assert force.shape[0] == self.num_envs, print('Actors should be merged.')
    #             has_collision = has_collision | (torch.norm(force, dim=1) < FORCE_THRESHOLD)
        
    #     self.scene.set_sim_state(init_state, env_idx=torch.where(has_collision)[0])
    #     return has_collision

    def collision_checker(self, new_cfg, active_env_idx):
        # TODO: make sure new_cfg is qpos
        qpos_to_set = self.agent.robot.get_qpos()
        qpos_to_set[active_env_idx] = new_cfg[active_env_idx]
        init_state = self.get_state_dict()
        self.agent.robot.set_qpos(qpos_to_set)
        action = np.zeros_like(self.agent.action_space.sample())
        if len(action.shape) == 1:
            action = action[None]
        self.step(action)
        if physx.is_gpu_enabled():
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()
        forces = self.agent.robot.get_net_contact_forces(list(self.agent.robot.links_map.keys()))
        has_collision = torch.linalg.norm(forces, dim=-1).sum(-1) > FORCE_THRESHOLD
        if has_collision.any():
            nonzero_env = [i.item() for i in has_collision.nonzero()]
            print(f'************ Collision detected for {nonzero_env}! **************')
        self.set_state_dict(init_state)
        return has_collision

    def uniform_cfg_sampler(self, active_env_idx, pad_value, active_joint_indices, pad_inactive_joints_value):
        qlimits = self.agent.robot.get_qlimits()  # (num_envs, 9, 2)
        
        # Separate the minimum and maximum limits
        q_min = qlimits[..., 0]  # Shape: (num_envs, num_joints)
        q_max = qlimits[..., 1]  # Shape: (num_envs, num_joints)

        # Select only the limits for the active joint indices
        active_q_min = q_min[..., active_joint_indices]  # Shape: (num_envs, len(active_joint_indices))
        active_q_max = q_max[..., active_joint_indices]  # Shape: (num_envs, len(active_joint_indices))
        
        # Initialize the output tensor with pad_value
        num_envs = qlimits.shape[0]
        num_joints = qlimits.shape[1]
        qpos = torch.full((num_envs, num_joints), pad_value, device=self.device)  # Shape: (num_envs, num_joints)

        # # Uniformly sample random numbers between 0 and 1 for active joints
        # random_samples = torch.rand(active_q_min.shape, device=self.device)  # Shape: (num_active_envs, len(active_joint_indices))
        
        if active_env_idx.dtype == torch.bool:
            active_env_true_idx = torch.nonzero(active_env_idx).squeeze(1)
            inactive_env_true_idx = torch.nonzero(~active_env_idx).squeeze(1)

        else:
            active_env_true_idx = active_env_idx
            all_env_idx = torch.arange(self.num_envs, device=active_env_idx.device)  # Shape: (self.num_envs,)
            # Find the inactive indices using set difference
            inactive_env_true_idx = torch.tensor(list(set(all_env_idx.cpu().numpy()) - set(active_env_true_idx.cpu().numpy())), device=self.device)
            
            # Optionally, sort the inactive indices
            inactive_env_true_idx = inactive_env_true_idx.sort().values

        
        # Uniformly sample random numbers between 0 and 1 for active joints in active environments
        random_samples = torch.rand((len(active_env_true_idx), len(active_joint_indices)), device=self.device)  # Shape: (num_active_envs, len(active_joint_indices))
        
        # Sample only for active environments
        sampled_qpos = active_q_min[active_env_true_idx] + random_samples * (active_q_max[active_env_true_idx] - active_q_min[active_env_true_idx])

        # Assign sampled values to the active joint indices for the active environments
        qpos[torch.tensor(active_env_true_idx, device=self.device).unsqueeze(1), torch.tensor(active_joint_indices, device=self.device)] = sampled_qpos
        
        # Create a mask for inactive joint indices
        inactive_joint_mask = torch.ones(num_joints, dtype=torch.bool, device=self.device)
        inactive_joint_mask[active_joint_indices] = False  # Mark active joints as False

        # Set inactive joint positions to pad_inactive_joints_value
        qpos[:, inactive_joint_mask] = pad_inactive_joints_value
        qpos[inactive_env_true_idx, ...] = pad_value
        
        return qpos


    def save_start_state(self, extra_state_dict):
        
        state_dict = self.get_state_dict()  # Only include states of actors and articulations
        state_dict.update(extra_state_dict)
        self.start_state = state_dict
        
    def load_start_state(self, start_state_dict):
        sys_dict = {}
        randomization_dict = {}
        for key, value in start_state_dict.items():
            if key == 'actors' or key == 'articulations':
                sys_dict[key] = value
            else:
                randomization_dict[key] = value

        self.set_state_dict(sys_dict)
        assert set(randomization_dict.keys()) == {'lighting', 'camera'}, 'Randomization dict should have only keys lighting and camera.'
        self._randomize_lighting(lighting=randomization_dict['lighting'])
        self._randomize_camera_pose(camera_pose=randomization_dict['camera'])
        