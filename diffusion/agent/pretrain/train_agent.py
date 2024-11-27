"""
Parent pre-training agent class.

"""

import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
from copy import deepcopy

from guided_dc.utils.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)
import GPUtil

DEVICE = "cuda"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda"):

    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


class EMA:
    """
    Empirical moving average

    """

    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PreTrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.num_gpus = torch.cuda.device_count()
        self.gpu_id = int(cfg.gpu_id)

        self.cfg = cfg

        # Wandb
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb and self.gpu_id == 0:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        if cfg.debug:
            if self.gpu_id == 0:
                torch.cuda.memory._record_memory_history(max_entries=100000)

        # Build model
        if self.num_gpus > 1:
            cfg.model.device = self.gpu_id
        self.model = hydra.utils.instantiate(cfg.model)

        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs.")
            print(self.gpu_id)
            from torch.utils.data.distributed import DistributedSampler
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            self.device = torch.device(f"cuda:{self.gpu_id}")
            self.ema = EMA(cfg.ema)
            self.ema_model = deepcopy(self.model.module)
            # dist.barrier()
        else:
            self.model = self.model.to(cfg.device)
            self.device = torch.device(cfg.device)
            self.ema = EMA(cfg.ema)
            self.ema_model = deepcopy(self.model)

        print(self.ema_model.device)
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            print(f"Allocated GPU memory after loading model: {allocated_memory/1024/1024/1024} GB")
        GPUtil.showUtilization(all=True)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.update_ema_freq = cfg.train.update_ema_freq
        self.epoch_start_ema = cfg.train.epoch_start_ema

        self.val_freq = cfg.train.get("val_freq", 100)
        self.val_batch_size = cfg.train.get("val_batch_size", self.batch_size)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        

        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            print(
                f"Allocated GPU memory after loading dataset: {allocated_memory/1024/1024/1024} GB"
            )
        GPUtil.showUtilization(all=True, useOldCode=False)

        if cfg.train.store_gpu:
            assert self.dataset_train.device != "cpu", self.dataset_train.device
            if self.num_gpus == 1:
                self.dataloader_train = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=False,
                )
            else:
                self.dataloader_train = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    num_workers=0,
                    shuffle=False,
                    pin_memory=False,
                    sampler=DistributedSampler(self.dataset_train),
                )
                log.info(f"Using distributed sampler")
            log.info(f"Using GPU memory for dataset")
        else:
            if self.num_gpus == 1:
                self.dataloader_train = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    num_workers=cfg.train.get("num_workers", 4),
                    shuffle=True,
                    pin_memory=True,
                    persistent_workers=cfg.train.get("persistent_workers", False),
                )
            else:
                self.dataloader_train = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    num_workers=cfg.train.get("num_workers", 4),
                    shuffle=False,
                    pin_memory=True,
                    persistent_workers=cfg.train.get("persistent_workers", False),
                    sampler=DistributedSampler(self.dataset_train),
                )
                log.info(f"Using distributed sampler")
            log.info(f"Using CPU memory for dataset")

        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            if cfg.train.store_gpu:
                assert self.dataset_val.device != "cpu", self.dataset_val.device
                if self.num_gpus == 1:
                    self.dataloader_val = torch.utils.data.DataLoader(
                        self.dataset_val,
                        batch_size=self.val_batch_size,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=False,
                    )
                else:
                    self.dataloader_val = torch.utils.data.DataLoader(
                        self.dataset_val,
                        batch_size=self.val_batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False,
                        sampler=DistributedSampler(self.dataset_val),
                    )
            else:
                if self.num_gpus == 1:
                    self.dataloader_val = torch.utils.data.DataLoader(
                        self.dataset_val,
                        batch_size=self.val_batch_size,
                        num_workers=cfg.train.get("num_workers", 4),
                        shuffle=True,
                        pin_memory=True,
                        persistent_workers=cfg.train.get("persistent_workers", False),
                    )
                else:
                    self.dataloader_val = torch.utils.data.DataLoader(
                        self.dataset_val,
                        batch_size=self.val_batch_size,
                        num_workers=cfg.train.get("num_workers", 4),
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=cfg.train.get("persistent_workers", False),
                        sampler=DistributedSampler(self.dataset_val),
                    )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

    def run(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.num_gpus > 1:
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        if self.num_gpus > 1:
            self.ema.update_model_average(self.ema_model, self.model.module)
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": (
                self.model.state_dict()
                if self.num_gpus == 1
                else self.model.module.state_dict()
            ),
            "ema": self.ema_model.state_dict(),
            # "cfg": self.cfg,
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.epoch = data["epoch"]
        if self.num_gpus == 1:
            self.model.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
        else:
            self.model.module.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
