"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

import threading
import cv2
import time
import numpy as np
from queue import Queue

from diffusion.agent.eval.eval_diffusion_agent_real import EvalDiffusionAgentReal


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
        cv2.imshow("Combined Viewer", combined_image)

        # Allow OpenCV to refresh the window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cv2.destroyAllWindows()


def process_obs(obs, obs_min, obs_max, ordered_obs_keys):
    """concatenate and normalize obs"""
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

    # Normalize
    obs = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
    obs = np.clip(obs, -1, 1)
    return obs


class EvalImgDiffusionAgentReal(EvalDiffusionAgentReal):

    def __init__(self, cfg):
        super().__init__(cfg)

        # visualize img observations
        self.visualize = cfg.get("visualize", False)

    def process_multistep_state(self, obs, prev_obs=None):
        if self.n_cond_step == 2:
            assert prev_obs
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
        ret = torch.from_numpy(ret).float().to(self.device)

        # TODO: use config
        # round gripper position
        ret[:, -1] = torch.round(ret[:, -1])
        return ret

    def process_multistep_img(self, obs, camera_indices, prev_obs=None, bgr2rgb=True):
        if self.n_img_cond_step == 2:  # TODO: better logic
            assert prev_obs
            images = {}
            for idx in camera_indices:
                image_1 = obs["image"][idx].transpose(2, 0, 1).copy()
                image_2 = prev_obs["image"][idx].transpose(2, 0, 1).copy()
                images[idx] = np.concatenate(
                    (
                        image_1[None, None],
                        image_2[None, None],
                    ),
                    axis=1,
                )
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()
                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
        else:
            images = {}

            for idx in camera_indices:
                image = obs["image"][idx].transpose(2, 0, 1).copy()
                images[idx] = image[None, None]
                if bgr2rgb:
                    images[idx] = images[idx][:, :, ::-1, :, :].copy()

                images[idx] = torch.from_numpy(images[idx]).to(self.device).float()
        return images

    def unnormalize_action(self, naction):
        action = (naction + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min
        return action

    def unnormalized_delta_action(self, naction, state):
        action = (naction[:, :-1] + 1) * (
            self.delta_max - self.delta_min + 1e-6
        ) / 2 + self.delta_min
        action += state[:, :-1]  # skip gripper
        gripper_action = (naction[:, -1:] + 1) * (
            self.action_max[-1] - self.action_min[-1] + 1e-6
        ) / 2 + self.action_min[-1]
        return np.concatenate([action, gripper_action], axis=-1)

    def unnormalize_obs(self, state):
        return (state + 1) * (self.obs_max - self.obs_min + 1e-6) / 2 + self.obs_min

    def run(self):
        if self.visualize:
            image_queues = {
                "Camera 1": Queue(),
                # "Camera 2": Queue(),
                "Camera 3": Queue(),
            }
            thread = threading.Thread(target=visualize_images, args=(image_queues,))
            thread.start()

        # Reset env before iteration starts
        self.model.eval()
        prev_obs = self.reset_env()
        obs = prev_obs
        np.set_printoptions(precision=3, suppress=True)
        camera_indices = ["2", "8"]

        # Check inference
        print("Warming up policy inference")
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs, prev_obs=prev_obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=camera_indices,
                prev_obs=prev_obs,
                bgr2rgb=True,
            )
            print(
                "RGB dimensions:",
                [cond["rgb"][idx].shape for idx in cond["rgb"].keys()],
            )
            print("State dimensions:", cond["state"].shape)
            samples = (
                self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
            )
            print(
                "Predicted action chunk dimensions:",
                samples.shape,
            )
            naction = samples[0, : self.act_steps]  # remove batch dimension
            prev_obs = obs
        if self.use_delta_actions:
            cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
            action = self.unnormalized_delta_action(naction, cur_state)
        else:
            action = self.unnormalize_action(naction)
        print("Action:", action)

        # Check images
        for i in camera_indices:
            plt.imshow(
                cond["rgb"][i][0, 0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            plt.savefig(f"image_{i}.png")
        input("Check images and then press anything to continue...")

        # TODO: some safety check making sure the sample predicted actions are not crazy

        # Run
        cond_states = []
        images = [[], []]
        actions = []
        robot_states = [
            np.concatenate(
                [
                    obs["robot_state"]["cartesian_position"],
                    [obs["robot_state"]["gripper_position"]],
                ]
            )
        ]
        try:
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Run policy inference with observations
                pre_obs_start_time = time.time()
                with torch.no_grad():
                    cond = {}
                    cond["state"] = self.process_multistep_state(
                        obs=obs,
                        prev_obs=prev_obs,
                    )
                    cond["rgb"] = self.process_multistep_img(
                        obs=obs,
                        camera_indices=camera_indices,
                        prev_obs=prev_obs,
                        bgr2rgb=True,
                    )
                    cond_states.append(cond["state"].cpu().numpy())
                    for i, k in enumerate(cond["rgb"].keys()):
                        images[i].append(cond["rgb"][k].cpu().numpy())

                    # Debug
                    if self.visualize:
                        for i, (window_name, image_queue) in enumerate(
                            image_queues.items()
                        ):
                            # Create a blank image and draw some text
                            image = images[i][-1][0, 0].transpose(1, 2, 0)
                            # cv2.putText(image, f"{window_name}: Frame {step}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Add the image to the corresponding queue
                            image_queue.put(image)
                    print(
                        "State: ",
                        self.unnormalize_obs(cond["state"].cpu().numpy()),
                    )
                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    # Run forward pass
                    samples = (
                        self.model(cond=cond, deterministic=True)
                        .trajectories.cpu()
                        .numpy()
                    )
                    naction = samples[0, : self.act_steps]  # remove batch dimension
                if self.use_delta_actions:
                    cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                    action = self.unnormalized_delta_action(naction, cur_state)
                else:
                    action = self.unnormalize_action(naction)
                actions.append(action)

                # Debug
                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")
                print("Action: ", action)

                # Run action chunk
                for a in action:
                    prev_obs = obs
                    self.env.step(a)
                    obs = self.env.get_observation()
                    robot_states.append(
                        np.concatenate(
                            [
                                obs["robot_state"]["cartesian_position"],
                                [obs["robot_state"]["gripper_position"]],
                            ]
                        )
                    )
                    time.sleep(0.06)

                # Debug
                step_end_time = time.time()
                print(f"Step time: {step_end_time - model_inf_end_time}")
        except KeyboardInterrupt:
            print("Interrupted by user")
        # finally:
        #     # Send exit signal to stop visualization
        #     for image_queue in image_queues.values():
        #         image_queue.put(None)

        # Save data
        np.savez_compressed(
            self.result_path,
            actions=np.array(actions),
            robot_states=np.array(robot_states),
            cond_states=np.array(cond_states),
            images=np.array(images),
        )
