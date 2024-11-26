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


def main():

    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", '-j', type=str, help="Job id, e.g., '100001'")
    parser.add_argument("--ckpt", '-c', type=int, help="ckpt id, e.g., 0")
    args = parser.parse_args()
    
    ckpt_path = '/home/lab/guided-data-collection/ckpts'
    
    # Get the config folder under the ckpt_path that starts with the job id, e.g., f'/home/lab/droid/ckpts/{job_id}_vit'
    job_id = args.job
    job_folder = [f for f in os.listdir(ckpt_path) if f.startswith(job_id)][0]
    job_folder = os.path.join(ckpt_path, job_folder)
    cfg_path = os.path.join(job_folder, 'config.yaml')
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(job_folder, f'state_{args.ckpt}.pt')

    # Initialize and run the agent
    cfg.gpu_id = 0
    cfg._target_ = 'diffusion.agent.eval.eval_diffusion_img_agent_real.EvalImgDiffusionAgentReal'
    cfg.model.network_path = ckpt_path
    
    if cfg.dataset_name.startswith('eefg'):
        cfg.ordered_obs_keys = ['cartesian_position', 'gripper_position']
    elif cfg.dataset_name.startswith('jsg'):
        cfg.ordered_obs_keys = ['joint_positions', 'gripper_position']
    else:
        raise NotImplementedError
    
    if '_eefg' in cfg.dataset_name:
        cfg.action_space = 'cartesian_position'
    elif '_jsg' in cfg.dataset_name:
        cfg.action_space = 'joint_position'
    else:
        raise NotImplementedError
    
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
