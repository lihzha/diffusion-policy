"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

# import pretty_errors
import logging
import math
import os
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import gymnasium as gym
from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode

from guided_dc.utils.io_utils import load_sim_hdf5_for_training

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


CKPT_PATH = "log/tomato_plate"


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", "-j", type=str, help="Job id, e.g., '100001'")
    parser.add_argument("--ckpt", "-c", type=int, help="ckpt id, e.g., 0")
    parser.add_argument(
        "--num_steps",
        "-s",
        type=int,
        help="Number of steps to run the agent",
        default=100,
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="guided_dc/cfg/simulation/pick_and_place.yaml",
    )
    parser.add_argument("--traj_path", "-tp", type=str)
    args = parser.parse_args()

    # Get the config folder under the ckpt_path that starts with the job id, e.g., f'/home/lab/droid/ckpts/{job_id}_vit'
    job_id = args.job
    job_folder = next(f for f in os.listdir(CKPT_PATH) if f.startswith(job_id))
    job_folder = os.path.join(CKPT_PATH, job_folder)
    cfg_path = os.path.join(job_folder, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(job_folder, f"checkpoint/state_{args.ckpt}.pt")

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
    
    traj, _ = load_sim_hdf5_for_training(args.traj_path)
    
    env_cfg = OmegaConf.load(args.env_config)
    OmegaConf.resolve(env_cfg)

    env_cfg.env.control_mode = cfg.action_space
    env_cfg.env.manip_obj.pos = [float(v) for v in traj["pick_obj_pos"]]
    env_cfg.env.manip_obj.rot = [float(v) for v in traj["pick_obj_rot"]]
    env_cfg.env.goal_obj.pos = [float(v) for v in traj["place_obj_pos"]]
    env_cfg.env.goal_obj.rot = [float(v) for v in traj["place_obj_rot"]]

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

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg, env=env)
    
    agent.act_steps = 8
    
    joint_positions = traj["observation/robot_state/joint_positions"]
    gripper_positions = traj["observation/robot_state/gripper_position"]
    images_0 = traj["observation/image/0"]
    images_2 = traj["observation/image/2"]
    actions = np.concatenate(
        [traj["action/joint_position"], traj["action/gripper_position"][:, None]],
        axis=-1,
    )
    # Process gripper actions
    gripper_positions = 1 - (gripper_positions / 0.04)
    gripper_positions = (gripper_positions > 0.2).astype(np.float32)

    np.set_printoptions(precision=3, suppress=True)
    
    gt_actions = []
    predicted_actions = []

    for i in range(len(joint_positions)-agent.act_steps):
        joint_position = joint_positions[i]
        gripper_position = gripper_positions[i]
        image_0 = images_0[i]
        image_2 = images_2[i]
        obs = {
            "robot_state": {
                "joint_positions": joint_position,
                "gripper_position": gripper_position,
            },
            "image": {
                "0": image_0,
                "2": image_2,
            },
        }
        
        # env.step(actions[i])
        # env.render()
        
        
        # gt_action = actions[i:i+agent.act_steps].squeeze()
        # print(gt_action)
        # predicted_action, _  = agent.run_single_step(obs)
        # predicted_action = predicted_action.squeeze()
        # gt_actions.append(gt_action)
        # predicted_actions.append(predicted_action)
        
        for j in range(100):
            _, new_obs = agent.run_single_step(obs)
            obs = new_obs
        break
        
        # predicted_action = agent.predict_single_step(
        #     obs, camera_indices=["0", "2"], bgr2rgb=False
        # ).squeeze()
    
    # gt_actions = np.stack(gt_actions, axis=0)
    # predicted_actions = np.stack(predicted_actions, axis=0)
    # print(gt_actions.shape, predicted_actions.shape)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    # for i in range(8):
    #     ax[i//4, i%4].plot(gt_actions[:,i], label="gt")
    #     ax[i//4, i%4].plot(predicted_actions[:,i], label="predicted")
    # plt.legend()
    # plt.savefig("actions.png")
    
    env.close()
    
        


if __name__ == "__main__":
    main()
