"""
Train diffusion policy

"""

import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)

import time
import GPUtil

from diffusion.agent.pretrain.train_agent import PreTrainAgent
from guided_dc.utils.timer import Timer
from guided_dc.utils.preprocess_utils import batch_apply


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        timer = Timer()
        self.epoch = 1
        for epoch in range(self.n_epochs):

            # multi-gpu chore
            if self.num_gpus > 1:
                import torch.distributed as dist

                dist.barrier()
                self.dataloader_train.sampler.set_epoch(epoch)

            # train
            loss_train_epoch = []
            cnt_batch = 0
            for batch_train in self.dataloader_train:
                batch_train = batch_apply(
                    batch_train, lambda x: x.to(self.device, non_blocking=True)
                )
                batch_train = batch_apply(batch_train, lambda x: x.float())

                # log.info("GPU used for loading batch:")
                # GPUtil.showUtilization(all=True)

                self.model.train()
                if self.num_gpus > 1:
                    loss_train = self.model.module.loss(*batch_train)
                else:
                    loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                with torch.no_grad():
                    self.dataloader_val.sampler.set_epoch(epoch)
                    self.model.eval()
                    for batch_val in self.dataloader_val:
                        batch_val = batch_apply(
                            batch_val, lambda x: x.to(self.device, non_blocking=True)
                        )
                        batch_val = batch_apply(batch_val, lambda x: x.float())
                        if self.num_gpus > 1:
                            loss_val = self.model.module.loss(*batch_val)
                        else:
                            loss_val = self.model.loss(*batch_val)
                        loss_val_epoch.append(loss_val.item())
                    self.model.train()
                    del batch_val
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if (
                self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs
            ) and self.gpu_id == 0:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0 and self.gpu_id == 0:

                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=True
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1

            if self.num_gpus > 1:
                dist.barrier()

            if (self.debug and self.epoch == 2) and self.gpu_id == 0:
                try:
                    torch.cuda.memory._dump_snapshot(
                        f"{self.cfg.train.batch_size}_mem_debug.pickle"
                    )
                except Exception as e:
                    logging.error(f"Failed to capture memory snapshot {e}")
                # Stop recording memory snapshot history.
                torch.cuda.memory._record_memory_history(enabled=None)
