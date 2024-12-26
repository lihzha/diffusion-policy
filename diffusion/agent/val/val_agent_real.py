"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import os

import numpy as np
import torch

from diffusion.agent.val.val_agent import ValAgent
from guided_dc.utils.preprocess_utils import batch_apply

log = logging.getLogger(__name__)


class ValAgentReal(ValAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
        self.normalization_stats_path = cfg.normalization_stats_path
        self.normalization_stats = np.load(
            self.normalization_stats_path, allow_pickle=True
        )
        self.obs_min = self.normalization_stats["obs_min"]
        self.obs_max = self.normalization_stats["obs_max"]
        self.action_min = self.normalization_stats["action_min"]
        self.action_max = self.normalization_stats["action_max"]

    def unnormalize_action(self, action):
        return (action + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min

    def unnormalize_obs(self, obs):
        return (obs + 1) * (self.obs_max - self.obs_min + 1e-6) / 2 + self.obs_min

    def normalize_action(self, action):
        return (
            2 * (action - self.action_min) / (self.action_max - self.action_min + 1e-6)
            - 1
        )

    @torch.no_grad()
    def run(self, num_steps=100):
        self.model.eval()
        losses = []

        # actions = np.load('/home/lab/droid/guided-data-collection/guided_dc/diffusion/log/real_pick_and_place/pick_and_place_eval_diffusion_unet_img_ta8_td100/2024-11-19_17-26-53_42/actions_79.npy')
        # cond_states = np.load('/home/lab/droid/guided-data-collection/guided_dc/diffusion/log/real_pick_and_place/pick_and_place_eval_diffusion_unet_img_ta8_td100/2024-11-19_17-26-53_42/cond_states_79.npy')
        # images = np.load('/home/lab/droid/guided-data-collection/guided_dc/diffusion/log/real_pick_and_place/pick_and_place_eval_diffusion_unet_img_ta8_td100/2024-11-19_17-26-53_42/images_79.npy')
        # robot_states = np.load('/home/lab/droid/guided-data-collection/guided_dc/diffusion/log/real_pick_and_place/pick_and_place_eval_diffusion_unet_img_ta8_td100/2024-11-19_17-26-53_42/robot_states_79.npy')

        np.set_printoptions(precision=3, suppress=True)
        # from collections import namedtuple
        # Batch = namedtuple("Batch", "actions conditions")

        predicted_action_unnorms = []
        true_action_unnorms = []

        for i, batch in enumerate(self.dataloader):
            # cond = {}
            # state = torch.from_numpy(cond_states[i]).to(self.device, non_blocking=True).float()
            # rgb = {}
            # for k in range(3):
            #     rgb[k] = torch.from_numpy(images[k, i]).to(self.device, non_blocking=True).float()

            # cond['state'] = state
            # cond['rgb'] = rgb
            # normalized_action = self.normalize_action(actions[i])
            # batch = Batch(conditions=cond, actions=torch.from_numpy(normalized_action[None,:]).to(self.device, non_blocking=True).float())

            batch = batch_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            batch = batch_apply(batch, lambda x: x.float())

            los = self.model.loss(*batch)
            losses.append(los.item())

            predicted_action = (
                self.model.sample(cond=batch.conditions, deterministic=True)[0]
                .cpu()
                .numpy()[:, : self.act_steps]
            )
            true_action = batch.actions.cpu().numpy()[:, : self.act_steps]

            predicted_action_unnorm = self.unnormalize_action(predicted_action)
            true_action_unnorm = self.unnormalize_action(true_action)
            # obs_unnorm = self.unnormalize_obs(batch.conditions["state"].cpu().numpy())

            predicted_action_unnorms.append(predicted_action_unnorm)
            true_action_unnorms.append(true_action_unnorm)

            # breakpoint()

            print(f"Batch {i} - Loss: {los.item()}")
            if i == num_steps:
                break

            # breakpoint()

        np.save(os.path.join(self.result_path, f"loss_{i}.npy"), losses)
        np.save(
            os.path.join(self.result_path, f"predicted_action_unnorm_{i}.npy"),
            predicted_action_unnorms,
        )
        np.save(
            os.path.join(self.result_path, f"true_action_unnorm_{i}.npy"),
            true_action_unnorms,
        )

        print(np.mean(losses))
