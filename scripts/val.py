"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys

# import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt


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


CKPT_PATH = "/home/lab/guided-data-collection/ckpts"

def main():

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", "-j", type=str, help="Job id, e.g., '100001'")
    parser.add_argument("--ckpt", "-c", type=int, help="ckpt id, e.g., 0")
    parser.add_argument("--num_steps", "-s", type=int, help="Number of steps to run the agent", default=100)
    args = parser.parse_args()

    # Get the config folder under the ckpt_path that starts with the job id, e.g., f'/home/lab/droid/ckpts/{job_id}_vit'
    job_id = args.job
    job_folder = [f for f in os.listdir(CKPT_PATH) if f.startswith(job_id)][0]
    job_folder = os.path.join(CKPT_PATH, job_folder)
    cfg_path = os.path.join(job_folder, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(job_folder, f"state_{args.ckpt}.pt")

    # Initialize and run the agent
    cfg.gpu_id = 0
    cfg._target_ = (
        "diffusion.agent.val.val_diffusion_img_agent_real.ValImgDiffusionAgentReal"
    )
    cfg.model.network_path = ckpt_path

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run(args.num_steps)
    
    # Visualize the results
    result_folder = agent.result_path
    loss = np.load(os.path.join(result_folder, f"loss_{args.num_steps}.npy"))
    predicted_action_unnorm = np.load(os.path.join(result_folder, f"predicted_action_unnorm_{args.num_steps}.npy"))
    true_action_unnorm = np.load(os.path.join(result_folder, f"true_action_unnorm_{args.num_steps}.npy"))
    plt.hist(loss, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.savefig(os.path.join(result_folder, f"loss_{args.num_steps}.png"))
    
    assert len(predicted_action_unnorm.shape) == 4 and predicted_action_unnorm.shape == true_action_unnorm.shape
    fig, axs = plt.subplots(4, 2, figsize=(10, 15))
    for i in range(predicted_action_unnorm.shape[-1]):
        error_i = np.abs(predicted_action_unnorm[..., i] - true_action_unnorm[..., i]).mean(axis=(1, 2))
        # axs[i//2, i%2].plot(error_i)
        axs[i//2, i%2].hist(error_i, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axs[i//2, i%2].set_title(f"Action dim {i}")
        # axs[i//2, i%2].set_xlabel("Step")
        axs[i//2, i%2].set_ylabel("Absolute Error (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"action_error_{args.num_steps}.png"))
  
if __name__ == "__main__":
    main()