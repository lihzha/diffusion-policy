import gymnasium as gym
import numpy as np
import sapien
import os
import logging
import hydra
from omegaconf import OmegaConf
import torch
import copy
import h5py
from datetime import datetime

from guided_dc.maniskill.mani_skill.envs.sapien_env import BaseEnv
from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.maniskill.mani_skill.utils.structs import Pose

from guided_dc.utils.io_utils import merge_rgb_array_videos
import guided_dc.envs.tasks
from guided_dc.utils.pose_utils import batch_transform_to_pos_quat
from guided_dc.utils.traj_utils import smooth_trajectories, get_waypoints, compute_delta_ee_pose_euler, interpolate_trajectory, process_paths
from guided_dc.utils.vis_utils import visualize_trajectory_as_video
from guided_dc.utils.hdf5_utils import save_extra_info_to_hdf5

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
            assert cfg.data_iter<=30, "Cannot save videos when collecting data for more than 30 rounds."
        # Join cfg.env_id, cfg.num_envs, cfg.data_iter, current date and time to create a unique trajectory name
        trajectory_name = f"{cfg.env_id}_numenvs{cfg.num_envs}_datarnd{cfg.data_iter}_seed{cfg.seed}" + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        env = RecordEpisode(
            env,
            record_dir,
            save_video=cfg.save_video,
            save_on_reset=True,
            info_on_video=False,
            save_trajectory=True,
            max_steps_per_video=env._max_episode_steps,
            record_env_state=False,
            trajectory_name=trajectory_name
        )

    # Log environment details if verbose output is enabled
    if not cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.unwrapped.control_mode}")
        log.info(f"Reward mode: {env.unwrapped.reward_mode}")
    
    if not cfg.render_mode == 'human':

        env._json_data['env_info']['env_kwargs']['init'] = OmegaConf.to_container(env._json_data['env_info']['env_kwargs']['init'])
        env._json_data['env_info']['env_kwargs']['rand'] = OmegaConf.to_container(env._json_data['env_info']['env_kwargs']['rand'])

    
    for iteration in range(cfg.data_iter):

        if iteration%20 == 1:
            obs, _ = env.reset(seed=cfg.seed, options={'reconfigure': True})
        else:
            obs, _ = env.reset(seed=cfg.seed)
        
        if iteration >= 1:
            save_extra_info_to_hdf5(env, {'success_envs': success_envs.cpu(), 'control_mode': env.unwrapped.control_mode, 'start_state': env.unwrapped.start_state})
        env.action_space.seed(cfg.seed)

        env.agent.set_control_mode('pd_ee_delta_pose', first_called=False)

        # Initialize the simulation
        for _ in range(10):
            action = np.zeros_like(env.agent.action_space.sample())
            if len(action.shape) == 1:
                action = action[None]
            obs, _, _, _, _ = env.unwrapped.step(action)

        qpos_to_set, qpos_paths_for_pull = env.unwrapped.get_target_qpos()

        from guided_dc.rrt.parallel_rrt import parallel_rrt_connect
        
        def get_extra_cfg():
            return env.unwrapped.agent.tcp.pose.raw_pose.clone()
        
        qpos_paths, _ = parallel_rrt_connect(start_cfg=env.agent.robot.get_qpos(), goal_cfg=qpos_to_set, iterations=cfg.rrt_iter, 
                                        sampler=env.uniform_cfg_sampler, step_size=0.1, collision_checker=env.unwrapped.collision_checker, 
                                        threshold=0.005, get_extra_cfg=None, start_ee=None, goal_ee=None)    # Shape: (num_envs, longest_traj_length, qpos_dim)

        if cfg.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = cfg.pause
                
        env.agent.set_control_mode('pd_joint_pos', first_called=False)

        qpos_paths = interpolate_trajectory(qpos_paths, 2)
        for i in range (cfg.smooth_iter):
            qpos_paths = smooth_trajectories(qpos_paths, window_size=5, batch_first=True)
            print(qpos_paths.size())

        # 1. Reaching the handle
        for i in range(qpos_paths.size(1)):
            # info_dict = {}
            action = qpos_paths[:, i, :-1]
            obs, rew, terminated, truncated, info = env.step(action)
            env.render()

        # 2. Closing the gripper
        # action = env.unwrapped.agent.robot.get_qpos().clone()[:, :-1]
        action[:, -1] = -1
        for i in range(20):
            obs, rew, terminated, truncated, info = env.step(action)
            env.render()
        
        success_envs = env.agent.is_grasping(env.unwrapped.handle_link)

        # 3. Pull out
        for i in range(qpos_paths_for_pull.size(1)):
            obs, rew, terminated, truncated, info = env.step(qpos_paths_for_pull[:, i])
            env.render() 

    obs, _ = env.reset(seed=cfg.seed)
    save_extra_info_to_hdf5(env, {'success_envs': success_envs.cpu(), 'control_mode': env.unwrapped.control_mode, 'start_state': env.unwrapped.start_state})
    env.close()
    
    if cfg.save_video:
        video_name = 'output'

        if record_dir and cfg.render_mode != "human":
            print(f"Saving video to {record_dir}")
            merge_rgb_array_videos(record_dir, name=video_name)


if __name__ == "__main__":
    main()
