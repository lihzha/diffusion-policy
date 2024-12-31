import logging
import math
import os
import re

import gymnasium as gym
import numpy as np
import sapien
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode

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


def read_config(file_path) -> dict:
    config_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            # Remove comments and strip whitespace
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Parse key-value pairs
            match = re.match(r"^(\w+):\s*(.*)$", line)
            if match:
                key, value = match.groups()

                # Convert value to appropriate type
                try:
                    if "[" in value and "]" in value:  # Handle lists
                        # # Add commas if missing between elements
                        # value = re.sub(
                        #     r"\s+", ",", value.strip("[")
                        # )  # Replace spaces with commas
                        value = eval(f"{value}")
                    elif "." in value:  # Handle floats
                        value = float(value)
                    else:  # Handle integers
                        value = int(value)
                except (ValueError, SyntaxError):
                    pass  # Keep value as string if conversion fails
                if value == "True":
                    value = True
                config_dict[key] = value

    return config_dict


def load_dataset(dataset_folder):
    dataset_path = os.path.join(dataset_folder, "dataset.npz")
    config_path = os.path.join(dataset_folder, "config.txt")
    norm_stats_path = os.path.join(dataset_folder, "norm.npz")

    config = read_config(config_path)
    # print(config)
    camera_indices = config["camera_indices"]
    # action_keys = config["action_keys"]
    # bservation_keys = config["observation_keys"]
    # img_resolution = config["img_resolution"]
    # bgr2rgb = config["bgr2rgb"]
    # num_traj = config["num_traj"]

    norm_stats = np.load(norm_stats_path)
    obs_min = norm_stats["action_min"]
    obs_max = norm_stats["action_max"]
    action_min = norm_stats["action_min"]
    action_max = norm_stats["action_max"]
    # delta_min = norm_stats["delta_min"]
    # delta_max = norm_stats["delta_max"]

    dataset = np.load(dataset_path, allow_pickle=True)
    images = dataset["images"].item()
    actions = dataset["actions"]
    states = dataset["states"]

    # 1. Check actions and states
    print("Image max:", [np.max(images[i]) for i in camera_indices])
    print("Image min:", [np.min(images[i]) for i in camera_indices])
    print("Image shape:", [images[i].shape for i in camera_indices])
    print("Action max:", np.max(actions, axis=0))
    print("Action min:", np.min(actions, axis=0))
    print("State max:", np.max(states, axis=0))
    print("State min:", np.min(states, axis=0))

    unnormed_actions = (actions + 1) * (action_max - action_min + 1e-6) / 2 + action_min
    unnormed_states = (states + 1) * (obs_max - obs_min + 1e-6) / 2 + obs_min
    print("Unnormed action max:", np.max(unnormed_actions, axis=0))
    print("Unnormed action min:", np.min(unnormed_actions, axis=0))
    print("Unnormed state max:", np.max(unnormed_states, axis=0))
    print("Unnormed state min:", np.min(unnormed_states, axis=0))

    env_cfg = OmegaConf.load("guided_dc/cfg/simulation/pick_and_place.yaml")
    OmegaConf.resolve(env_cfg)
    env_cfg.env.control_mode = "pd_joint_pos"
    env_cfg.env.render_mode = "human"
    env_cfg.env.shader = "default"
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

    env.reset()

    if render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = env_cfg.pause

    unnormed_actions[:, -1] = 1 - 2 * unnormed_actions[:, -1]
    for i in range(len(unnormed_actions)):
        if unnormed_actions[i, -1] < 0:
            unnormed_actions[i, -1] = -1
        else:
            unnormed_actions[i, -1] = 1

    for i in range(10000):
        env.step(unnormed_actions[i])
        env.render()

    env.close()

    # fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    # for i in range(8):
    #     ax[i // 2, i % 2].plot(states[:, i], label=f"State {i}")
    #     ax[i // 2, i % 2].plot(actions[:, i], label=f"Action {i}")
    #     ax[i // 2, i % 2].legend()
    # plt.savefig(os.path.join(dataset_folder, "actions_states.png"))

    # # 2. Visualize images
    # k = 1000
    # stack_videos_horizontally(
    #     *[images[i][k : k + 200].transpose(0, 2, 3, 1) for i in camera_indices],
    #     os.path.join(dataset_folder, "images.mp4"),
    #     bgr2rgb=bgr2rgb,
    # )


if __name__ == "__main__":
    dataset_folder = "data/jsg_jsg_2cam_192__sim_1.0"
    load_dataset(dataset_folder)
