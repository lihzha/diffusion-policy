import os
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from sapien.render import RenderMaterial
from transforms3d.euler import euler2quat


class TabletopSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, robot_init_qpos_noise=0.02):
        super().__init__(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.cfg = cfg
        self.table_model_file = cfg.table.model_file
        self.floor_texture_file = cfg.floor_texture_file
        self.robot_init_qpos = cfg.robot_init_qpos
        self.robot_init_pos = cfg.robot_init_pos
        self.robot_init_euler = cfg.robot_init_rot
        self.table_height = cfg.table.table_height

        # Instead of moving the table, move the robot and the background
        self.robot_init_pos = np.array(self.robot_init_pos) + np.array(
            [0, 0, -self.table_height]
        )

        if os.path.isdir(self.table_model_file):
            table_model_files = [
                f for f in os.listdir(self.table_model_file) if f.endswith(".obj")
            ]
            self.table_model_file = os.path.join(
                self.table_model_file, np.random.choice(table_model_files)
            )
        if os.path.isdir(self.floor_texture_file):
            floor_texture_files = [
                f
                for f in os.listdir(self.floor_texture_file)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
            self.floor_texture_file = os.path.join(
                self.floor_texture_file, np.random.choice(floor_texture_files)
            )

    def build_background(self):
        builder = self.scene.create_actor_builder()
        p = np.array(self.cfg.background.pos) + np.array([0, 0, -self.table_height])
        background_pose = sapien.Pose(
            p=p,
            # p=[0.07, -0.28, -0.12],
            q=euler2quat(*self.cfg.background.rot),
        )
        builder.add_visual_from_file(
            filename=self.cfg.background.model_file,
            scale=[1] * 3,
            pose=background_pose,
        )
        builder.initial_pose = background_pose
        background = builder.build_kinematic(name="background")
        self.scene_objects.append(background)

    def build(self):
        # 1. Build the table
        builder = self.scene.create_actor_builder()
        table_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        # Always set table scale to 1
        builder.add_nonconvex_collision_from_file(
            filename=self.table_model_file,
            scale=[1.0] * 3,
            pose=table_pose,
        )
        # builder.add_box_collision(  # TODO: Table-specific, but potentially faster.
        #     pose=sapien.Pose(p=[0, 0, -0.025]),
        #     half_size=(0.7, 0.3575, 0.025),
        # )

        builder.add_visual_from_file(
            filename=self.table_model_file,
            scale=[1.0] * 3,
            pose=table_pose,
            material=self._get_table_material(),
        )
        builder.initial_pose = table_pose
        table = builder.build_kinematic(name="table-workspace")
        # aabb = (
        #     table._objs[0]
        #     .find_component_by_type(sapien.render.RenderBodyComponent)
        #     .compute_global_aabb_tight()
        # )
        # self.table_length = aabb[1, 0] - aabb[0, 0]
        # self.table_width = aabb[1, 1] - aabb[0, 1]
        # self.table_height = aabb[1, 2] - aabb[0, 2]
        # print(self.table_length, self.table_width)
        # breakpoint()
        # self.table_thickness = self.cfg.table.thickness

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500

        # 2. Build the ground
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-self.cfg.table.thickness - self.cfg.table.leg_length,
            texture_file=self.floor_texture_file,
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

        # 3. Build the background
        self.build_background()

        # 4. Build distractor objects
        # self.build_distractor()

    # def build_distractor(self):
    #     for cfg in self.cfg.distractor:
    #         if cfg.type == "objaverse":
    #             lvis_annotations = objaverse.load_lvis_annotations()
    #             uids_to_load = lvis_annotations[cfg.obj_name]
    #             # processes = multiprocessing.cpu_count()
    #             obj_dicts = objaverse.load_objects(
    #                 uids=uids_to_load, download_processes=1
    #             )
    #             asset_path_list = list(obj_dicts.values())
    #             rand_idx = torch.randperm(len(asset_path_list))
    #             rand_idx = [0, 1, 2, 3]
    #             model_files = [asset_path_list[idx] for idx in rand_idx]
    #             model_files = np.concatenate(
    #                 [model_files]
    #                 * np.ceil(self.env.num_envs / len(asset_path_list)).astype(int)
    #             )[: self.env.num_envs]
    #             if (
    #                 self.env.num_envs > 1
    #                 and self.env.num_envs < len(asset_path_list)
    #                 and self.env.reconfiguration_freq <= 0
    #             ):
    #                 print(
    #                     """There are less parallel environments than total available models to sample.
    #                     Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
    #                     or set reconfiguration_freq to be > 1."""
    #                 )
    #         elif cfg.type == "custom":
    #             model_files = [cfg.model_file] * self.env.num_envs
    #         else:
    #             raise ValueError(f"Unknown object type for distractors: {cfg.type}")
    #         _objs: List[Actor] = []
    #         for i, model_file in enumerate(model_files):
    #             # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
    #             builder = self.scene.create_actor_builder()
    #             builder.set_scene_idxs([i])

    #             distractor_pose = sapien.Pose(
    #                 p=cfg.pos,
    #                 q=euler2quat(*cfg.rot),
    #             )

    #             builder.add_nonconvex_collision_from_file(
    #                 filename=model_file,
    #                 scale=cfg.scale,
    #                 pose=distractor_pose,
    #                 # material=None,
    #             )
    #             builder.add_visual_from_file(
    #                 filename=model_file,
    #                 scale=cfg.scale,
    #                 pose=distractor_pose,
    #                 # material=None,
    #             )
    #             builder.initial_pose = distractor_pose

    #             _objs.append(
    #                 builder.build_kinematic(name=f"distractor-{cfg.obj_name}-{i}")
    #             )
    #             self.scene.remove_from_state_dict_registry(_objs[-1])
    #         distractor = Actor.merge(_objs, name=f"distractor-{cfg.obj_name}")
    #         self.scene.add_to_state_dict_registry(distractor)
    #         aabb = (
    #             distractor._objs[0]
    #             .find_component_by_type(sapien.render.RenderBodyComponent)
    #             .compute_global_aabb_tight()
    #         )
    #         length = aabb[1, 0] - aabb[0, 0]
    #         w = aabb[1, 1] - aabb[0, 1]
    #         h = aabb[1, 2] - aabb[0, 2]
    #         print(length, w, h)
    #         self.scene_objects.append(distractor)

    def _get_table_material(self):
        from sapien.render import RenderTexture2D

        table_material = RenderMaterial()
        table_material.base_color_texture = RenderTexture2D(
            filename=self.cfg.table.material_file,
            srgb=True,
        )
        table_material.diffuse_texture = RenderTexture2D(
            filename=self.cfg.table.material_file,
            srgb=True,
        )
        # table_material.roughness_texture = RenderTexture2D(
        #     filename=self.cfg.table.material_file,
        #     srgb=False,
        # )
        # Emission texture
        table_material.roughness = 0.2  # Very rough, like paper
        # table_material.metallic = 0.2
        # Specular reflection (minimal for paper)
        # table_material.specular = 0.6  # Low specular reflection
        return table_material

    def initialize(self, env_idx: torch.Tensor):
        """
        Initialize the environment with the specified table and robot configuration.

        Coordinate System:
        - When visulized, x-axis is red, y-axis is green, and z-axis is blue.
        - The table's frame is aligned with the global frame, where its top surface defines z=0.
        - The length of the table is aligned with the x-axis, and width is aligned with the y-axis.
        - The original robot's frame is also aligned with the global frame, and rotated by π/4
          ('+' stands for counter-clockwise) afterwards during episode initialization.

        - The robot is placed on the table (z=0), with random (x,y) positions according to `robot_pos`
          passed to the function, which is generated by a domain randomization function.
        - `qpos` represents the initial joint angles of the 7 joints and the initial opening extent
          of the 2 gripper fingers.
        """

        b = len(env_idx)
        self.table.set_pose(sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)))
        if self.env.robot_uids in ["panda", "panda_wristcam", "panda_wristcam_irom"]:
            qpos = np.array(self.robot_init_qpos)
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose(p=self.robot_init_pos, q=euler2quat(*self.robot_init_euler))
            )
            # self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            ("panda", "panda"),
            ("panda_wristcam", "panda_wristcam"),
        ]:
            assert (
                isinstance(self.robot_init_pos[0], list)
                and isinstance(self.robot_init_euler[0], list)
                and len(self.robot_init_pos) == 2
                and len(self.robot_init_euler) == 2
            ), "robot_init_pos and robot_init_euler should be lists of length 2"
            agent: MultiAgent = self.env.agent
            qpos = np.array(self.robot_init_qpos)
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose(
                    p=self.robot_init_pos[0], q=euler2quat(*self.robot_init_euler[0])
                )
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose(
                    p=self.robot_init_pos[1], q=euler2quat(*self.robot_init_euler[1])
                )
            )
        else:
            raise NotImplementedError("Unknown robot_uids")
