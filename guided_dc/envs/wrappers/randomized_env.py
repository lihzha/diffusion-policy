import torch
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from pathlib import Path

from mani_skill.envs.utils import randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor

from guided_dc.utils.randomization import randomize_continuous, randomize_by_percentage
from guided_dc.utils.pose_utils import quaternion_multiply
from guided_dc.utils.obj_utils import get_obj_asset

class RandEnv(BaseEnv):
    def __init__(self, *args, robot_uids="panda", rand_cfg=None, num_envs=1, reconfiguration_freq=None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.rand_cfg = rand_cfg
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 0
            else:
                reconfiguration_freq = 0
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
        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100
            )
        ]

    @property
    def _default_viewer_camera_configs(
        self,
    ) -> CameraConfig:
        """Default configuration for the viewer camera, controlling shader, fov, etc. By default if there is a human render camera called "render_camera" then the viewer will use that camera's pose."""
        return CameraConfig(uid="viewer", pose=sapien.Pose([0, 0, 1]), width=1920, height=1080, shader_pack="default", near=0.0, far=1000, mount=self.cam_mount)


    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=sapien.Pose(), width=512, height=512, fov=1, near=0.01, far=100, mount=self.cam_mount
        )


    def _randomize_lighting(self):
        shadow = self.enable_shadow
        ambient_min = self.rand_cfg.lighting.ambient_min
        ambient_max = self.rand_cfg.lighting.ambient_max
        ambient_light = randomize_continuous(min=ambient_min, max=ambient_max, size=3)
        self.scene.set_ambient_light(ambient_light)
        #TODO: how to remove previous directional lights
        # self.scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        # )
        # self.scene.add_directional_light([0, 0, -1], [1, 1, 1])
    
    def _randomize_camera_pose(self):
        rad_range = self.rand_cfg.camera.rad_range
        pose = sapien_utils.look_at(eye=[1.5, 0, 1], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-rad_range, rad_range)
            ),
        )
        self.cam_mount.set_pose(pose)

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
        q_delta_zrot_max = self.rand_cfg.object.manip_obj_delta_zrot_max
        q_delta_zrot_min = self.rand_cfg.object.manip_obj_delta_zrot_min
        q_delta_zrot = randomize_continuous(q_delta_zrot_min, q_delta_zrot_max, 1)
        manip_obj_q = quaternion_multiply(q, euler2quat(0, 0, q_delta_zrot))
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
            q_delta_zrot = randomize_continuous(q_delta_zrot_min, q_delta_zrot_max, 1)

            goal_obj_q = quaternion_multiply(q, euler2quat(self.init_cfg.goal.raw, self.init_cfg.goal.yaw, q_delta_zrot))
            goal_obj.set_pose(Pose.create_from_pq(p=goal_obj_xyz, q=goal_obj_q))

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
