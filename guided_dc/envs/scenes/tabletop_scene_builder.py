import os
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from transforms3d.euler import euler2quat


# TODO (stao): make the build and initialize api consistent with other scenes
class TabletopSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, robot_init_qpos_noise=0.02):
        super().__init__(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.table_model_file = cfg.table_model_file
        self.floor_texture_file = cfg.floor_texture_file
        self.table_scale = cfg.table_scale
        self.robot_init_qpos = cfg.robot_init_qpos
        self.robot_init_pos = cfg.robot_init_pos
        self.robot_init_euler = cfg.robot_init_rot

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
        background_file = "guided_dc/assets/background/background_full.obj"
        builder = self.scene.create_actor_builder()
        background_pose = sapien.Pose(
            p=[0.04, -0.39, -0.105],
            # p=[0.07, -0.28, -0.12],
            q=euler2quat(0, 0, 0),
        )
        builder.add_visual_from_file(
            filename=background_file,
            scale=[1] * 3,
            pose=background_pose,
        )
        builder.initial_pose = background_pose
        background = builder.build_kinematic(name="background")
        self.scene_objects.append(background)

    def build(self):
        builder = self.scene.create_actor_builder()
        table_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        builder.add_nonconvex_collision_from_file(
            filename=self.table_model_file,
            scale=[self.table_scale] * 3,
            pose=table_pose,
        )

        # builder.add_box_collision(         # TODO: Table-specific, but potentially faster.
        #     pose=sapien.Pose(p=[0, 0, 2+0.05/2-0.1]),
        #     half_size=(4 / 2, 2 / 2, 0.05 / 2),
        # )

        builder.add_visual_from_file(
            filename=self.table_model_file,
            scale=[self.table_scale] * 3,
            pose=table_pose,
            material=self._get_table_material(),
        )
        builder.initial_pose = table_pose

        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        print(self.table_length, self.table_width)

        # TODO: Table-specific
        self.table_thickness = 0.05 * self.table_scale

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-self.table_height,
            texture_file=self.floor_texture_file,
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

        self.build_background()

    def _get_table_material(self):
        # Create a material object
        from sapien.render import RenderMaterial, RenderTexture2D

        table_material = RenderMaterial()
        table_material.base_color_texture = RenderTexture2D(
            filename="guided_dc/assets/table/material_0.png",
            srgb=True,
        )
        table_material.diffuse_texture = RenderTexture2D(
            filename="guided_dc/assets/table/material_0.png",
            srgb=True,
        )
        # table_material.roughness_texture = RenderTexture2D(
        #     filename="guided_dc/assets/table/material_0.png",
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
