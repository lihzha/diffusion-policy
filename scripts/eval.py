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

import torch
import signal
import torch.distributed as dist

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


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "guided_dc/cfg/real/pick_and_place"
    ),  # possibly overwritten by --config-path
    config_name="diffusion_unet_vit.yaml",
)
def main(cfg: OmegaConf):

    # Initialize and run the agent
    cfg['gpu_id'] = 0
    cfg['_target_'] = 'agent.eval.eval_diffusion_img_agent_real.EvalImgDiffusionAgentReal'
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()



if __name__ == "__main__":
    main()
