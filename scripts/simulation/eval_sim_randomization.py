"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import argparse
import datetime
import logging
import math
import os
import sys

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.utils.io_utils import dict_to_omegaconf_format, merge_rgb_array_videos

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

CKPT_PATH = "ckpts/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        help="Job id, e.g., '100001'",
        default="1056700",
    )
    parser.add_argument(
        "--ckpt",
        "-c",
        type=int,
        help="ckpt id, e.g., 0",
        default=200,
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="guided_dc/cfg/simulation/pick_and_place.yaml",
    )
    args = parser.parse_args()

    # Get the config folder under the ckpt_path that starts with the job id, e.g., f'/home/lab/droid/ckpts/{job_id}_vit'
    job_id = args.job
    job_folder = next(f for f in os.listdir(CKPT_PATH) if f.startswith(job_id))
    job_folder = os.path.join(CKPT_PATH, job_folder)
    cfg_path = os.path.join(job_folder, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(job_folder, f"state_{args.ckpt}.pt")

    # add datetime to logdir
    cfg.logdir = os.path.join(
        cfg.logdir, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Initialize and run the agent
    cfg.gpu_id = 0
    cfg._target_ = "diffusion.agent.eval.eval_agent_sim.EvalAgentSim"
    cfg.model.network_path = ckpt_path

    # Set up control and proprio
    if cfg.dataset_name.startswith("eefg"):
        cfg.ordered_obs_keys = ["cartesian_position", "gripper_position"]
    elif cfg.dataset_name.startswith("jsg"):
        cfg.ordered_obs_keys = ["joint_positions", "gripper_position"]
    else:
        raise NotImplementedError
    if "_eefg" in cfg.dataset_name:
        cfg.action_space = "pd_ee_pose"
    elif "_jsg" in cfg.dataset_name:
        cfg.action_space = "pd_joint_pos"
    else:
        raise NotImplementedError

    # Initialize environment
    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    env_cfg = OmegaConf.load(args.env_config)
    OmegaConf.resolve(env_cfg)

    if not env_cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(env_cfg)}")

    env_cfg.env.control_mode = cfg.action_space

    env = gym.make(env_cfg.env.env_id, cfg=env_cfg.env)

    output_dir = env_cfg.record.output_dir
    render_mode = env_cfg.env.render_mode
    if output_dir and render_mode != "human":
        log.info(f"Recording environment episodes to: {output_dir}")
        env = RecordEpisode(
            env,
            max_steps_per_video=env._max_episode_steps,
            **env_cfg.record,
        )

    # Log environment details if verbose output is enabled
    if not env_cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.unwrapped.control_mode}")
        log.info(f"Reward mode: {env.unwrapped.reward_mode}")

    if (not render_mode == "human") and (env_cfg.record.save_trajectory):
        env._json_data["env_info"]["env_kwargs"] = OmegaConf.create(
            env._json_data["env_info"]["env_kwargs"]
        )
        env._json_data["env_info"]["env_kwargs"] = OmegaConf.to_container(
            env._json_data["env_info"]["env_kwargs"]
        )

    # env.reset()

    # if render_mode is not None:
    #     viewer = env.render()
    #     if isinstance(viewer, sapien.utils.Viewer):
    #         viewer.paused = env_cfg.pause

    cfg.n_steps = 120
    cfg.act_steps = 6

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg, env)

    initial_positions = []

    x_range = np.linspace(-0.4, -0.2, 3)
    y_range = np.linspace(-0.2, 0.2, 5)
    for x in x_range:
        for y in y_range:
            for xx in x_range:
                for yy in y_range:
                    if x == xx and y == yy:
                        continue
                    initial_positions.append(
                        (
                            [x, y, 0.0, -1.57265, -0.0464526, 3.0965],
                            [xx, yy, 0.01, 1.569, 2.9, -0.001066],
                        )
                    )

    for rnd in range(len(initial_positions)):
        for trial in range(5):
            p = initial_positions[rnd]
            env_cfg.env.manip_obj.pos = [float(v) for v in p[0][:3]]
            env_cfg.env.manip_obj.rot = [float(v) for v in p[0][3:]]
            env_cfg.env.goal_obj.pos = [float(v) for v in p[1][:3]]
            env_cfg.env.goal_obj.rot = [float(v) for v in p[1][3:]]
            # lighting = env_cfg.env.lighting[rnd]
            # env.unwrapped.set_ambient_light(lighting)

            env_states = agent.run()
            print(env_states)
            env.flush_video()

            if output_dir and render_mode != "human":
                video_dir = f"{job_id}/{args.ckpt}"
                video_dir = os.path.join(output_dir, video_dir)
                os.makedirs(video_dir, exist_ok=True)
                video_dir = os.path.join(
                    video_dir, f"{p[0][0]}_{p[0][1]}_{p[1][0]}_{p[1][1]}"
                )
                os.makedirs(video_dir, exist_ok=True)

                # Save the video
                video_name = (
                    f"{env_states['success']}_{trial}.mp4".replace("[", "")
                    .replace("]", "")
                    .replace(" ", "_")
                )
                video_name = os.path.join(video_dir, video_name)
                print(f"Saving video to {video_name}")
                merge_rgb_array_videos(output_dir, video_name)

                if trial == 0:
                    # Save the env_states to a cfg file at the same location as the video
                    states_name = "env_states.yaml"
                    states_name = os.path.join(video_dir, states_name)
                    with open(states_name, "w") as f:
                        OmegaConf.save(
                            OmegaConf.create(dict_to_omegaconf_format(env_states)), f
                        )

                    # Save env_cfg to a yaml file at the same location as the video
                    cfg_name = "config.yaml"
                    cfg_name = os.path.join(video_dir, cfg_name)
                    with open(cfg_name, "w") as f:
                        OmegaConf.save(env_cfg, f)

    env.close()


if __name__ == "__main__":
    main()
