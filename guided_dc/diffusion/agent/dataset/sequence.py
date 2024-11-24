"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

import numpy as np
import torch
import logging
import pickle
import random

from guided_dc.utils.preprocess_utils import preprocess_img, Batch

log = logging.getLogger(__name__)

class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
        store_gpu=False,
        use_delta_actions=True
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.use_delta_actions = use_delta_actions

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
            images = dataset["images"]
            if images.dtype == np.dtype('O'):
                images = images.item()
                if len(images.keys()) > 1:
                    self.use_multi_images = True
                elif len(images.keys()) == 1:
                    self.use_multi_images = False
                else:
                    raise ValueError("No images found in the dataset")
            else:
                self.use_multi_images = False
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)            
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps, self.cond_steps)

        if store_gpu:
        # Extract states and actions up to max_n_episodes
            self.states = (
                torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
            )  # (total_num_steps, obs_dim)
            self.actions = (
                torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
            )  # (total_num_steps, action_dim)
        else:
            self.states = dataset["states"][:total_num_steps].astype(np.float32)
            self.actions = dataset["actions"][:total_num_steps].astype(np.float32)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        if self.use_img:
            if not self.use_multi_images:
                if store_gpu:
                    self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                        device
                    )  # (total_num_steps, C, H, W)
                else:
                    self.images = dataset["images"][:total_num_steps]
                log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")
            else:
                self.images = {}
                for idx in images.keys():
                    if store_gpu:
                        self.images[idx] = torch.from_numpy(images[idx][:total_num_steps]).to(
                            device
                        )
                    else:
                        self.images[idx] = images[idx][:total_num_steps]
                log.info(f"Loading multiple images from {len(self.images)} cameras")
                log.info(f"Images shape/type: {self.images[idx].shape, self.images[idx].dtype}")
        
        self.store_gpu = store_gpu

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        if self.store_gpu:
            actions = self.actions[start:end].clone()
        else:
            actions = self.actions[start:end].copy()
        if self.use_delta_actions:
            actions[1:] = actions[1:] - actions[0]
        
        
        stack_pkg = torch if self.store_gpu else np

        states = stack_pkg.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        
        conditions = {"state": states}

        if self.use_img:
            if not self.use_multi_images:
                images = self.images[(start - num_before_start) : end]
                images = stack_pkg.stack(
                    [
                        preprocess_img(images[max(num_before_start - t, 0)])
                        for t in reversed(range(self.img_cond_steps))
                    ]
                )
            else:
                images = {}
                for idx in self.images.keys():
                    images[idx] = self.images[idx][(start - num_before_start) : end]
                    images[idx] = stack_pkg.stack(
                        [
                            preprocess_img(images[idx][max(num_before_start - t, 0)])
                            for t in reversed(range(self.img_cond_steps))
                        ]
                    )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps, cond_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            min_start = cur_traj_index + cond_steps - 1
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(min_start, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [ind for ind in self.indices if ind not in train_indices]
        self.set_indices(train_indices)
        return val_indices
    
    def set_indices(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

