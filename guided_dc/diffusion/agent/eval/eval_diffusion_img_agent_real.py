"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from agent.eval.eval_diffusion_agent_real import EvalDiffusionAgentReal
from guided_dc.utils.preprocess_utils import preprocess_img

import threading
import cv2
import time
import numpy as np
from queue import Queue

# Function to visualize multiple images in a single window
def visualize_images(image_queues):
    while True:
        frames = []
        for image_queue in image_queues.values():
            if not image_queue.empty():
                image = image_queue.get()
                if image is None:  # Exit signal
                    return
                frames.append(image)
            else:
                # If no new frame, add a blank placeholder
                frames.append(255 * np.ones((300, 300, 3), dtype=np.uint8))

        # Combine frames horizontally or vertically
        combined_image = cv2.hconcat(frames)  # Combine side-by-side
        # combined_image = cv2.vconcat(frames)  # Combine top-to-bottom
        
        # Display the combined image in one window
        cv2.imshow('Combined Viewer', combined_image)

        # Allow OpenCV to refresh the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()

def process_obs(obs, obs_min, obs_max, ordered_obs_keys):

    obs = np.concatenate(
        [
            (
                obs["robot_state"][ordered_obs_keys[i]]
                if ordered_obs_keys[i] != "gripper_position"
                else [obs["robot_state"][ordered_obs_keys[i]]]
            )
            for i in range(len(ordered_obs_keys))
        ]
    )[None, :]

    obs = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
    obs = np.clip(obs, -1, 1)
    obs[0,3] = np.abs(obs[0,3])
    return obs


class EvalImgDiffusionAgentReal(EvalDiffusionAgentReal):

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
        self.ordered_obs_keys = cfg.ordered_obs_keys
        
        self.visualize = getattr(cfg, "visualize", False)

    def process_multistep_obs(self, obs, prev_obs=None):
        if prev_obs is not None:
            assert self.n_cond_step == 2
            ret = np.stack(
                (
                    process_obs(
                        prev_obs, self.obs_min, self.obs_max, self.ordered_obs_keys
                    ),
                    process_obs(obs, self.obs_min, self.obs_max, self.ordered_obs_keys),
                ),
                axis=1,
            )
        else:
            assert self.n_cond_step == 1
            ret = process_obs(obs, self.obs_min, self.obs_max, self.ordered_obs_keys)
        return torch.from_numpy(ret).float().to(self.device)

    def process_multistep_img(self, obs, prev_obs=None):

        camera_idx = ['0', '2', '3']
        if prev_obs is not None:
            assert self.n_cond_step == 2
            images = {}
            for idx in camera_idx:
                resized_images_1 = preprocess_img(obs["image"][idx].transpose(2, 0, 1))
                resized_images_2 = preprocess_img(
                    prev_obs["image"][idx].transpose(2, 0, 1)
                )

                images[idx] = np.concatenate(
                    (
                        resized_images_1[None, :][None, :],
                        resized_images_2[None, :][None, :],
                    ),
                    axis=1,
                )
                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                assert images[idx].shape == (1, 2, 3, 240, 320), images[idx].shape
        else:
            assert self.n_cond_step == 1
            images = {}
            for idx in camera_idx:
                resized_images = obs["image"][idx].transpose(2, 0, 1)
                images[idx] = preprocess_img(resized_images)[None, :][None, :]
                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
                assert images[idx].shape == (1, 1, 3, 240, 320), images[idx].shape
        return images

    def postprocess_action(self, output):
        action = output[:, : self.act_steps]
        action = (action + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min
        return action.squeeze(0)

    def unnormalize_obs(self, state):
        return (state + 1) * (
            self.obs_max - self.obs_min + 1e-6
        ) / 2 + self.obs_min

    def run(self):
        
        if self.visualize:

            image_queues = {
                "Camera 1": Queue(),
                "Camera 2": Queue(),
                "Camera 3": Queue(),
                }
            thread = threading.Thread(target=visualize_images, args=(image_queues,))
            thread.start()

        # Reset env before iteration starts
        self.model.eval()
        prev_obs = self.reset_env()
        obs = prev_obs.copy()
        self.env.camera_reader.set_trajectory_mode()

        np.set_printoptions(precision=3, suppress=True)

        print("Waiting for realsense")
        time.sleep(1.0)

        print("Warming up policy inference")
        with torch.no_grad():
            cond = {}

            cond["state"] = self.process_multistep_obs(obs=obs, prev_obs=prev_obs)
            cond["rgb"] = self.process_multistep_img(obs=obs, prev_obs=prev_obs)

            print(cond['rgb'].keys())
            print(cond['rgb']['0'].shape, cond['rgb']['2'].shape, cond['rgb']['3'].shape)
            print(cond["state"].shape)

            samples = self.model(cond=cond, deterministic=True)
            output = samples.trajectories.cpu().numpy()  # n_env x horizon x act

            prev_obs = obs.copy()

        action = self.postprocess_action(output)
        print("Ready!")

        

        # Collect a set of trajectories from env
        cond_states = []
        images = [[], [], []]
        actions = []
        robot_states = [np.concatenate([obs["robot_state"]["cartesian_position"].copy(), [obs["robot_state"]["gripper_position"]]])]


        try:
            for step in range(self.n_steps):
                # timer = Timer()
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                pre_obs_start_time = time.time()

                with torch.no_grad():
                    cond = {}

                    cond["state"] = self.process_multistep_obs(obs=obs, prev_obs=prev_obs)
                    cond["rgb"] = self.process_multistep_img(obs=obs, prev_obs=prev_obs)

                    cond_states.append(cond["state"].cpu().numpy().copy())
                    for i, k in enumerate(cond["rgb"].keys()):
                        images[i].append(cond["rgb"][k].cpu().numpy().copy())

                    if self.visualize:
                        for i, (window_name, image_queue) in enumerate(image_queues.items()):
                            # Create a blank image and draw some text
                            image = images[i][-1][0, 0].copy().transpose(1, 2, 0)
                            # cv2.putText(image, f"{window_name}: Frame {step}", (20, 150),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Add the image to the corresponding queue
                            image_queue.put(image)


                    print("State: ", self.unnormalize_obs(cond["state"].cpu().numpy().copy()))
                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    samples = self.model(cond=cond, deterministic=True)
                    output = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                action = self.postprocess_action(output)

                actions.append(action.copy())

                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")

                print("Action: ", action)
                # Apply multi-step action
                for idx, a in enumerate(action):
                    # if a[6] >= 0.5:
                    #     a[6] = 1
                    # else:
                    #     a[6] = 0
                    
                    self.env.step(a)
                    obs = self.env.get_observation()
                    if idx == len(action) - 2:
                        prev_obs = obs.copy()
                    robot_states.append(np.concatenate([obs["robot_state"]["cartesian_position"].copy(), [obs["robot_state"]["gripper_position"]]]))
                    # time.sleep(1/13.5)
                    time.sleep(0.06)

                step_end_time = time.time()
                print(f"Step time: {step_end_time - model_inf_end_time}")
                # print(obs['robot_state']['cartesian_position'][:3])
        except KeyboardInterrupt:
            print("Interrupted by user") 
        # finally:
        #     # Send exit signal to stop visualization
        #     for image_queue in image_queues.values():
        #         image_queue.put(None)

        np.save(f"cond_states_{step}.npy", np.array(cond_states))
        np.save(f"images_{step}.npy", np.array(images))
        np.save(f"actions_{step}.npy", np.array(actions))
        np.save(f"robot_states_{step}.npy", np.array(robot_states))