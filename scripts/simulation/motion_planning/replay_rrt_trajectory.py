import logging
import os
from datetime import datetime

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.utils.io_utils import merge_rgb_array_videos
from guided_dc.utils.traj_utils import load_traj_to_replay

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
                cfg.data_iter <= 30
            ), "Cannot save videos when collecting data for more than 30 rounds."
        # Join cfg.env_id, cfg.num_envs, cfg.data_iter, current date and time to create a unique trajectory name
        trajectory_name = (
            f"{cfg.env_id}_numenvs{cfg.num_envs}_datarnd{cfg.data_iter}_seed{cfg.seed}"
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

    start_state, actions, control_mode = load_traj_to_replay(
        cfg.traj_path, cfg.traj_idx
    )
    env.load_start_state(start_state)

    assert (
        control_mode == env.unwrapped.control_mode
    ), "Control mode of the trajectory does not match the control mode of the environment"

    for action in actions:
        obs, rew, terminated, truncated, info = env.step(action)
        env.render()

    env.close()

    if cfg.save_video:
        video_name = "traj_replay"

        if record_dir and cfg.render_mode != "human":
            print(f"Saving video to {record_dir}")
            merge_rgb_array_videos(record_dir, name=video_name)


if __name__ == "__main__":
    main()
