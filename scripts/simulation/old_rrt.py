import copy
import logging
import os
from datetime import datetime

import gymnasium as gym
import hydra
import numpy as np
import sapien
import torch
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf

from guided_dc.rrt.parallel_rrt import parallel_rrt_connect
from guided_dc.utils.io_utils import merge_rgb_array_videos
from guided_dc.utils.traj_utils import (
    interpolate_trajectory,
    smooth_trajectories,
)

OmegaConf.register_resolver("pi_div", lambda x: np.pi / float(x))

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "guided_dc/cfg"
    ),  # possibly overwritten by --config-path
    config_name="rrt_config",
    version_base=None,
)
def main(cfg):
    OmegaConf.resolve(cfg)

    if not cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(cfg)}")

    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    # Check if we are rendering in a single scene
    parallel_in_single_scene = cfg.render_mode == "human"
    if cfg.render_mode == "human" and cfg.obs_mode in [
        "sensor_data",
        "rgb",
        "rgbd",
        "depth",
        "point_cloud",
    ]:
        log.info(
            "Disabling parallel single scene/GUI render as observation mode is a visual one. "
            "Change observation mode to 'state' or 'state_dict' to enable parallel rendering."
        )
        parallel_in_single_scene = False

    if cfg.render_mode == "human" and cfg.num_envs == 1:
        parallel_in_single_scene = False

    # assert cfg.control_mode == 'pd_joint_pos', "Only support pd_joint_pos control mode"
    # Create the environment using the loaded configuration
    env = gym.make(
        cfg.env_id,
        obs_mode=cfg.obs_mode,
        reward_mode=cfg.reward_mode,
        control_mode=cfg.control_mode,
        render_mode=cfg.render_mode,
        sensor_configs=dict(shader_pack=cfg.shader),
        human_render_camera_configs=dict(shader_pack=cfg.shader),
        viewer_camera_configs=dict(shader_pack=cfg.shader),
        num_envs=cfg.num_envs,
        sim_backend=cfg.sim_backend,
        parallel_in_single_scene=parallel_in_single_scene,
        robot_uids="panda_wristcam",
        **cfg[cfg.env_id],
    )

    # Check if recording is enabled and adjust the environment accordingly
    record_dir = cfg.record_dir
    if record_dir and cfg.render_mode != "human":
        record_dir = record_dir.format(env_id=cfg.env_id)
        log.info(f"Recording environment episodes to: {record_dir}")
        if cfg.save_video:
            assert (
                cfg.data_rnd <= 30
            ), "Cannot save videos when collecting data for more than 30 rounds."
        # Join cfg.env_id, cfg.num_envs, cfg.data_rnd, current date and time to create a unique trajectory name
        trajectory_name = (
            f"{cfg.env_id}_numenvs{cfg.num_envs}_datarnd{cfg.data_rnd}_seed{cfg.seed}"
            + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        env = RecordEpisode(
            env,
            record_dir,
            save_video=cfg.save_video,
            save_on_reset=True,
            info_on_video=False,
            save_trajectory=True,
            max_steps_per_video=env._max_episode_steps,
            record_env_state=False,
            trajectory_name=trajectory_name,
        )

    # Log environment details if verbose output is enabled
    if not cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.unwrapped.control_mode}")
        log.info(f"Reward mode: {env.unwrapped.reward_mode}")

    env._json_data["env_info"]["env_kwargs"]["init"] = OmegaConf.to_container(
        env._json_data["env_info"]["env_kwargs"]["init"]
    )
    env._json_data["env_info"]["env_kwargs"]["rand"] = OmegaConf.to_container(
        env._json_data["env_info"]["env_kwargs"]["rand"]
    )

    for iteration in range(cfg.data_rnd):
        if iteration % 20 == 1:
            obs, _ = env.reset(seed=cfg.seed, options={"reconfigure": True})
        else:
            obs, _ = env.reset(seed=cfg.seed)
        if iteration >= 1:
            traj_id = "traj_{}".format(env._episode_id)
            group = env._h5_file[traj_id]
            # group.create_dataset("success_envs", data=success_envs.cpu())
        env.action_space.seed(cfg.seed)

        env.agent.set_control_mode("pd_ee_delta_pose", first_called=False)

        # Initialize the simulation
        for _ in range(10):
            action = np.zeros_like(env.agent.action_space.sample())
            if len(action.shape) == 1:
                action = action[None]
            obs, _, _, _, _ = env.unwrapped.step(action)

        goal_cfg_at_global: sapien.Pose = copy.deepcopy(
            env.get_grasp_pose()
        )  # in global frame
        q0 = env.agent.robot.get_qpos().clone()

        goal_cfg_at_base = env.agent.robot.pose.inv() * goal_cfg_at_global
        goal_qpos_for_arm = env.agent.controller.controllers[
            "arm"
        ].kinematics.compute_ik(
            target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False
        )
        goal_qpos_for_gripper = torch.ones_like(goal_qpos_for_arm)[
            :, :2
        ]  # 2 dim, one gripper hand mimic the other so just set them the same
        qpos_to_set = torch.cat((goal_qpos_for_arm, goal_qpos_for_gripper), dim=-1)

        pull_direction: torch.Tensor = env.unwrapped.handle_link_normal.clone()
        goal_pose_for_pull = Pose.create_from_pq(
            p=goal_cfg_at_global.p + pull_direction * 0.2, q=goal_cfg_at_global.q
        )
        goal_pose_for_pull_at_base = env.agent.robot.pose.inv() * goal_pose_for_pull
        goal_qpos_for_pull_arm = env.agent.controller.controllers[
            "arm"
        ].kinematics.compute_ik(
            target_pose=goal_pose_for_pull_at_base, q0=q0, use_delta_ik_solver=False
        )
        goal_qpos_for_pull_gripper = torch.ones_like(goal_qpos_for_pull_arm)[:, :1] * (
            -1
        )
        qpos_to_set_pull_start_arm = env.agent.controller.controllers[
            "arm"
        ].kinematics.compute_ik(
            target_pose=goal_cfg_at_base, q0=q0, use_delta_ik_solver=False
        )
        qpos_to_set_pull_start = torch.cat(
            (qpos_to_set_pull_start_arm, goal_qpos_for_pull_gripper), dim=-1
        )
        qpos_to_set_for_pull = torch.cat(
            (goal_qpos_for_pull_arm, goal_qpos_for_pull_gripper), dim=-1
        )
        qpos_paths_for_pull = interpolate_trajectory(
            torch.cat(
                (
                    qpos_to_set_pull_start.unsqueeze(1),
                    qpos_to_set_for_pull.unsqueeze(1),
                ),
                dim=1,
            ),
            10,
        )

        def get_extra_cfg():
            return env.agent.tcp.pose.raw_pose.clone()

        qpos_paths, ee_global_absolute_paths = parallel_rrt_connect(
            start_cfg=env.agent.robot.get_qpos(),
            goal_cfg=qpos_to_set,
            iterations=300,
            sampler=env.uniform_cfg_sampler,
            step_size=0.1,
            collision_checker=env.unwrapped.collision_checker,
            threshold=0.005,
            get_extra_cfg=get_extra_cfg,
            start_ee=get_extra_cfg(),
            goal_ee=goal_cfg_at_global.raw_pose.clone(),
        )  # Shape: (num_envs, longest_traj_length, qpos_dim)

        # visualize_ee_traj(ee_global_absolute_paths[0])

        if cfg.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = cfg.pause

        env.agent.set_control_mode("pd_joint_pos", first_called=False)

        qpos_paths = interpolate_trajectory(qpos_paths, 2)
        qpos_paths = smooth_trajectories(qpos_paths, window_size=5, batch_first=True)

        # for _ in range(cfg.smooth_rnd):
        #     qpos_paths = smooth_trajectories(qpos_paths, window_size=10, batch_first=True)

        # actions = []
        # observations = []
        # rewards = []
        # terminateds = []
        # truncates = []
        # infos = []
        # ee_poses = []
        # qposes = []
        # qvels = []
        # env_states = []
        # goal_cfgs = []
        # controller_states = []
        # root_poses = []
        # 1. Reaching the handle
        for i in range(qpos_paths.size(1)):
            # info_dict = {}
            action = qpos_paths[:, i, :-1]
            obs, rew, terminated, truncated, info = env.step(action)

            # actions.append(action.clone())
            # observations.append(copy.deepcopy(obs))
            # rewards.append(rew.clone())
            # terminateds.append(terminated.clone())
            # truncates.append(truncated.clone())
            # infos.append(copy.deepcopy(info))
            # ee_poses.append(env.unwrapped.agent.tcp.pose.raw_pose.clone())
            # qposes.append(env.unwrapped.agent.robot.get_qpos().clone())
            # qvels.append(env.unwrapped.agent.robot.get_qvel().clone())
            # env_states.append(copy.deepcopy(env.unwrapped.get_state_dict()))
            # goal_cfgs.append(goal_cfg_at_global.raw_pose.clone())
            # controller_states.append(copy.deepcopy(env.unwrapped.agent.controller.get_state()))
            # root_poses.append(env.unwrapped.agent.robot.pose.raw_pose.clone())

            env.render()

        # 2. Closing the gripper
        action[:, -1] = -1
        for _ in range(20):
            obs, rew, terminated, truncated, info = env.step(action)

            # actions.append(action.clone())
            # observations.append(copy.deepcopy(obs))
            # rewards.append(rew.clone())
            # terminateds.append(terminated.clone())
            # truncates.append(truncated.clone())
            # infos.append(copy.deepcopy(info))
            # ee_poses.append(env.unwrapped.agent.tcp.pose.raw_pose.clone())
            # qposes.append(env.unwrapped.agent.robot.get_qpos().clone())
            # qvels.append(env.unwrapped.agent.robot.get_qvel().clone())
            # env_states.append(copy.deepcopy(env.unwrapped.get_state_dict()))
            # goal_cfgs.append(goal_cfg_at_global.raw_pose.clone())
            # controller_states.append(copy.deepcopy(env.unwrapped.agent.controller.get_state()))
            # root_poses.append(env.unwrapped.agent.robot.pose.raw_pose.clone())

            env.render()

        success_envs = env.agent.is_grasping(env.unwrapped.handle_link)

        # 3. Pull out
        for i in range(qpos_paths_for_pull.size(1)):
            obs, rew, terminated, truncated, info = env.step(qpos_paths_for_pull[:, i])

            # actions.append(action.clone())
            # observations.append(copy.deepcopy(obs))
            # rewards.append(rew.clone())
            # terminateds.append(terminated.clone())
            # truncates.append(truncated.clone())
            # infos.append(copy.deepcopy(info))
            # ee_poses.append(env.unwrapped.agent.tcp.pose.raw_pose.clone())
            # qposes.append(env.unwrapped.agent.robot.get_qpos().clone())
            # qvels.append(env.unwrapped.agent.robot.get_qvel().clone())
            # env_states.append(copy.deepcopy(env.unwrapped.get_state_dict()))
            # goal_cfgs.append(goal_cfg_at_global.raw_pose.clone())
            # controller_states.append(copy.deepcopy(env.unwrapped.agent.controller.get_state()))
            # root_poses.append(env.unwrapped.agent.robot.pose.raw_pose.clone())

            env.render()

        # actions = torch.stack(actions, dim=0)
        # rewards = torch.stack(rewards, dim=0)
        # terminateds = torch.stack(terminateds, dim=0)
        # truncates = torch.stack(truncates, dim=0)
        # ee_poses = torch.stack(ee_poses, dim=0)
        # qposes = torch.stack(qposes, dim=0)
        # qvels = torch.stack(qvels, dim=0)
        # goal_cfgs = torch.stack(goal_cfgs, dim=0)
        # root_poses = torch.stack(root_poses, dim=0)

        # # Add the tensor gpu trajectory data to a hdf5 file
        # import h5py
        # with h5py.File('traj_data.hdf5', 'w') as f:
        #     f.create_dataset('actions', data=actions.cpu().numpy())
        #     f.create_dataset('observations', data=observations)
        #     f.create_dataset('rewards', data=rewards.cpu().numpy())
        #     f.create_dataset('terminateds', data=terminateds.cpu().numpy())
        #     f.create_dataset('truncates', data=truncates.cpu().numpy())
        #     f.create_dataset('ee_poses', data=ee_poses.cpu().numpy())
        #     f.create_dataset('qposes', data=qposes.cpu().numpy())
        #     f.create_dataset('qvels', data=qvels.cpu().numpy())
        #     f.create_dataset('env_states', data=env_states.cpu().numpy())
        #     f.create_dataset('goal_cfgs', data=goal_cfgs.cpu().numpy())
        #     f.create_dataset('root_poses', data=root_poses.cpu().numpy())
        #     f.create_dataset('controller_states', data=controller_states)
        #     f.create_dataset('info', data=infos)
        #     f.create_dataset('env_states', data=env_states)
        #     f.create_dataset('success_envs', data=success_envs.cpu().numpy())
        #     f.close()

        # if cfg.replay_with_controller:
        #     print('*'*80)
        #     # 1. Convert qpos to ee pose tractories.
        #     ee_pose_batch = env.agent.controller.controllers['arm'].kinematics.pk_chain.forward_kinematics(qpos_paths.reshape(-1, qpos_paths.size(-1)), end_only=True)
        #     ee_base_pose_paths = batch_transform_to_pos_quat(ee_pose_batch.get_matrix()).reshape(qpos_paths.size(0), qpos_paths.size(1), -1)  # (num_envs, longest_traj_length, 7)
        #     visualize_trajectory_as_video(ee_global_absolute_paths[0], "end_effector_trajectory.mp4")

        #     # ee_pose_paths = interpolate_trajectory(ee_global_absolute_paths, 3)
        #     print(ee_base_pose_paths.shape)

        #     # 2. Smooth ee pose trajectories.
        #     ee_base_pose_paths = interpolate_trajectory(ee_base_pose_paths, 3)
        #     smooth_ee_base_paths = ee_base_pose_paths
        #     for _ in range(cfg.smooth_rnd):
        #         smooth_ee_base_paths = smooth_trajectories(smooth_ee_base_paths, window_size=5, batch_first=True) # Shape: (num_envs, longest_traj_length, ee_dim)
        #     # smooth_ee_paths = ee_pose_paths
        #     # from mani_skill.utils.structs import Pose
        #     # ee_base_poses = []
        #     # for i in range(smooth_ee_paths.size(1)):
        #     #     ee_global_pose = smooth_ee_paths[:, i, :]
        #     #     ee_pose = Pose.create_from_pq(p=ee_global_pose[:, :3], q=ee_global_pose[:, 3:])
        #     #     ee_base_pose = env.agent.robot.pose.inv() * ee_pose
        #     #     ee_base_poses.append(ee_base_pose.raw_pose.clone())
        #     # ee_base_poses = torch.stack(ee_base_poses, dim=1)
        #     # 3. Get delta ee pose trajectories
        #     delta_ee_paths = compute_delta_ee_pose_euler(smooth_ee_base_paths)

        #     # 4. Run with pd_ee_delta_pose_controller
        #     assert cfg.control_mode == 'pd_ee_delta_pose', cfg.control_mode
        #     for i in range(delta_ee_paths.size(1)):
        #         action_arm = delta_ee_paths[:, i]
        #         action_gripper = torch.ones_like(action_arm)[:, :1]
        #         action = torch.cat((action_arm, action_gripper), dim=-1)
        #         obs, _, _, _, _ = env.step(action)
        #         env.render()
        #     # prev_pos = env.agent.tcp.pose.raw_pose[:,:3].clone()
        #     # for i in range(100):
        #     #     action = np.zeros_like(env.action_space.sample())
        #     #     action[:, 1] = 0.1
        #     #     env.step(action)
        #     #     print(env.agent.tcp.pose.raw_pose[:,:3]-prev_pos)
        #     #     prev_pos = env.agent.tcp.pose.raw_pose[:,:3].clone()
        #     #     env.render()

        # else:

        #     # Reaching the handle
        #     for i in range(max_traj_len):

        #         env.agent.robot.set_qpos(qpos_paths[:,i,:])
        #         # env.agent.robot.set_qvel(torch.zeros_like(qpos_to_set))
        #         # env.agent.scene._gpu_apply_all()
        #         # env.agent.scene.px.gpu_update_articulation_kinematics()
        #         # env.agent.scene._gpu_fetch_all()
        #         env.step(np.zeros_like(env.action_space.sample()))
        #         ee_paths.append(copy.deepcopy(env.agent.tcp.pose.raw_pose))
        #         # env.render()

        #     env.scene.set_sim_state(init_state)

        #     ee_paths = torch.stack(ee_paths, dim=1)
        #     smooth_ee_paths = ee_paths
        #     for i in range(cfg.smooth_rnd):
        #         smooth_ee_paths = smooth_trajectories(smooth_ee_paths, window_size=10, batch_first=True)
        #     processed_ee_paths, pad_start_idx = process_paths(smooth_ee_paths, goal_cfg_at_global.raw_pose.clone())
        #     # processed_ee_paths[:,-1] = goal_cfg_at_global.raw_pose.clone()

        #     smooth_ee_paths = interpolate_trajectory(processed_ee_paths, 5)

        #     from mani_skill.utils.structs import Pose
        #     ee_base_poses = []
        #     for i in range(smooth_ee_paths.size(1)):
        #         ee_global_pose = smooth_ee_paths[:, i, :]
        #         ee_pose = Pose.create_from_pq(p=ee_global_pose[:, :3], q=ee_global_pose[:, 3:])
        #         ee_base_pose = env.agent.robot.pose.inv() * ee_pose
        #         ee_base_poses.append(ee_base_pose.raw_pose.clone())
        #     ee_base_poses = torch.stack(ee_base_poses, dim=1)
        #     # 3. Get delta ee pose trajectories
        #     delta_ee_paths = compute_delta_ee_pose_euler(ee_base_poses)

        #     # visualize_trajectory_as_video(smooth_ee_paths[0])

        #     for i in range(delta_ee_paths.size(1)):

        #         action_arm = delta_ee_paths[:, i]
        #         action_gripper = torch.ones_like(action_arm)[:, :1] * 10
        #         action = torch.cat((action_arm, action_gripper), dim=-1)
        #         obs, _, _, _, _ = env.step(action)
        #         env.render()

        #     # # Closing the gripper
        #     # action = torch.zeros_like(ee_base_poses[:, 0, :])
        #     # action[:, -1] = -0.1
        #     # for i in range(10):
        #     #     obs, _, _, _, _ = env.step(action)
        #     #     # input("enter2")
        #     #     env.render()

        #     # # Pull out
        #     # current_ee_pose: torch.Tensor = env.agent.tcp.pose.raw_pose.clone()
        #     # pull_direction: torch.Tensor = env.handle_link_normal.clone()
        #     # delta_ee_pose = torch.zeros_like(current_ee_pose)
        #     # pull_steps = 40
        #     # delta_ee_pose[:, :3] = pull_direction * (1/pull_steps)

        #     # for i in range(pull_steps):
        #     #     obs, _, _, _, _ = env.step(delta_ee_pose)
        #     #     env.render()

        # env.reset(seed=cfg.seed)
        # # Replay the trajectory from the hdf5 file
        # with h5py.File('traj_data.hdf5', 'r') as f:
        #     actions = torch.tensor(f['actions'])
        #     observations = torch.tensor(f['observations'])
        #     rewards = torch.tensor(f['rewards'])
        #     terminateds = torch.tensor(f['terminateds'])
        #     truncates = torch.tensor(f['truncates'])
        #     ee_poses = torch.tensor(f['ee_poses'])
        #     qposes = torch.tensor(f['qposes'])
        #     qvels = torch.tensor(f['qvels'])
        #     env_states = torch.tensor(f['env_states'])
        #     goal_cfgs = torch.tensor(f['goal_cfgs'])
        #     root_poses = torch.tensor(f['root_poses'])
        #     controller_states = torch.tensor(f['controller_states'])
        #     infos = f['info']
        #     env_states = f['env_states']
        #     success_envs = torch.tensor(f['success_envs'])

        # env.set_state_dict(env_states[0])
        # env.agent.set_control_mode('pd_joint_pos', first_called=False)
        # for i in range(actions.size(0)):
        #     obs, _, _, _, _ = env.step(actions[i])
        #     env.render()

    obs, _ = env.reset(seed=cfg.seed)
    traj_id = "traj_{}".format(env._episode_id)
    group = env._h5_file[traj_id]
    group.create_dataset("success_envs", data=success_envs.cpu())
    env.close()

    if cfg.save_video:
        video_name = "output"

        if record_dir and cfg.render_mode != "human":
            print(f"Saving video to {record_dir}")
            merge_rgb_array_videos(record_dir, name=video_name)


if __name__ == "__main__":
    main()
