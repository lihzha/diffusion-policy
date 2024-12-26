"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import threading
import time
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.agent.eval.eval_agent import EvalAgent

log = logging.getLogger(__name__)


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


class EvalAgentReal(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg, env=None)

        # visualize img observations
        self.visualize = cfg.get("visualize", False)

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
                self.model.sample(cond=cond, deterministic=True)
                .trajectories.cpu()
                .numpy()
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
                        for i, (_, image_queue) in enumerate(image_queues.items()):
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
                        self.model.sample(cond=cond, deterministic=True)
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
