import logging
import os

import gymnasium as gym
import hydra
import numpy as np
import sapien
from mani_skill.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf

from guided_dc.utils.io_utils import merge_rgb_array_videos

OmegaConf.register_resolver("pi_div", lambda x: np.pi / float(x))

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "guided_dc/cfg"
    ),  # possibly overwritten by --config-path
    config_name="config",
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
        **cfg[cfg.env_id],
    )

    # Check if recording is enabled and adjust the environment accordingly
    record_dir = cfg.record_dir
    if record_dir and cfg.render_mode != "human":
        record_dir = record_dir.format(env_id=cfg.env_id)
        log.info(f"Recording environment episodes to: {record_dir}")
        env = RecordEpisode(
            env,
            record_dir,
            info_on_video=False,
            save_trajectory=False,
            max_steps_per_video=env._max_episode_steps,
        )

    # Log environment details if verbose output is enabled
    if not cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.unwrapped.control_mode}")
        log.info(f"Reward mode: {env.unwrapped.reward_mode}")

    obs, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    # Initialize the simulation
    for _ in range(5):
        action = np.zeros_like(env.action_space.sample())
        obs, _, _, _, _ = env.step(action)

    if cfg.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = cfg.pause
    for rnd in range(cfg.rnd):
        i = 0

        while i < 100000:
            action = env.action_space.sample()
            # xy_obj = env.get_wrapper_attr('obj').pose.raw_pose.clone()[0][:2].numpy()
            # xy_agent = env.get_wrapper_attr('agent').robot.get_pose().raw_pose.clone()[0][:2].numpy()
            # action = np.array([0,0]+[0., 0., 0., 0., 0.])[None]
            obs, reward, terminated, truncated, info = env.step(action)
            if not cfg.quiet:
                print("reward", reward)
                print("terminated", terminated)
                print("truncated", truncated)
                print("info", info)
            if cfg.render_mode is not None:
                env.render()
            # Never breaks for rendering purpose
            if cfg.render_mode is None:
                if (terminated | truncated).any():
                    break
            i += 1
        print("*" * 20)
        print(f"Round {rnd}")
        obs, _ = env.reset()

        # Initialize the simulation
        for _ in range(8):
            action = np.zeros_like(env.action_space.sample())
            obs, _, _, _, _ = env.step(action)

    env.close()

    if record_dir and cfg.render_mode != "human":
        print(f"Saving video to {record_dir}")
        merge_rgb_array_videos(record_dir)


if __name__ == "__main__":
    main()
