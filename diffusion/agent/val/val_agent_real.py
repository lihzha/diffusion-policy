"""
Parent eval agent class.

"""

import os
import numpy as np
import torch
import hydra
import logging
import random

log = logging.getLogger(__name__)



class ValAgentReal:

    def __init__(self, cfg):
        super().__init__()
        
        
        self.cfg = cfg
        self.gpu_id = int(cfg.gpu_id)
        self.device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Eval params
        self.n_steps = cfg.n_steps

        self.dataset = hydra.utils.instantiate(cfg.train_dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            num_workers=4 if self.dataset.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset.device == "cpu" else False,
        )

        self.batch_size = cfg.batch_size
        
    def run(self):
        pass
