"""
Evaluate pre-trained/DPPO-fine-tuned pixel-based diffusion policy.

"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.agent.eval.eval_agent import EvalAgent
from guided_dc.utils.pose_utils import quaternion_to_euler_xyz

log = logging.getLogger(__name__)


def update(env_states, env_success, terminated, info, eval_step=0):
    new_env_success = torch.logical_or(env_success, terminated)
    env_states["success"] = new_env_success.cpu().numpy().copy()
    for i in range(len(new_env_success)):
        if new_env_success[i] != env_success[i]:
            env_states["eval_steps"][i] = eval_step
            env_states["env_elapsed_steps"][i] = info["elapsed_steps"].cpu().numpy()[i]
    env_success = new_env_success
    return env_states, env_success


def step_without_action(env):
    info = env.get_info()
    obs = env.get_obs(info)
    return (
        obs,
        info,
    )


class EvalAgentSim(EvalAgent):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # self.dataset = np.load(
        #     "data/jsg_jsg_2cam_192__sim_1.0/dataset.npz", allow_pickle=True
        # )
        # self.states = self.dataset["states"]
        # self.images = self.dataset["images"].item()
        # self.actions = self.dataset["actions"]
        # self.action_timestep = list(range(0, len(self.actions), self.act_steps))
        # self.state_timestep = list(range(0, len(self.states), self.act_steps))

    def predict_single_step(self, obs, camera_indices=["0", "2"], bgr2rgb=False):
        self.model.eval()
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=camera_indices,
                bgr2rgb=bgr2rgb,
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
            naction = samples[:, : self.act_steps]  # remove batch dimension
            if self.use_delta_actions:
                cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                action = self.unnormalized_sim_delta_action(naction, cur_state)
                print("using delta")
            else:
                action = self.unnormalize_action(naction)
            return action

    def run_single_step(self, obs, camera_indices=["0", "2"], bgr2rgb=False):
        self.model.eval()
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=camera_indices,
                bgr2rgb=bgr2rgb,
            )
            # print(
            #     "RGB dimensions:",
            #     [cond["rgb"][idx].shape for idx in cond["rgb"].keys()],
            # )
            # print("State dimensions:", cond["state"].shape)
            samples = (
                self.model.sample(cond=cond, deterministic=True)
                .trajectories.cpu()
                .numpy()
            )
            # print(
            #     "Predicted action chunk dimensions:",
            #     samples.shape,
            # )
            naction = samples[:, : self.act_steps]  # remove batch dimension
            if self.use_delta_actions:
                cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                action = self.unnormalized_sim_delta_action(naction, cur_state)
                print("using delta")
            else:
                action = self.unnormalize_action(naction)
            action = self.postprocess_sim_gripper_action(action)

            print("Action:", action)

            # Run action chunk
            for action_step in range(self.act_steps):
                a = action[:, action_step]

                # a = np.concatenate([a, a[..., -1:]], axis=1)
                # a[..., -2:] = (a[..., -2:] + 1) * 0.02
                # self.env.agent.robot.set_qpos(a)
                # obs, info = step_without_action(self.env)
                # terminated = torch.tensor(0).to(torch.bool)

                new_obs, _, _, _, _ = self.env.step(a)

                self.env.render()
                new_obs = self.process_sim_observation(new_obs)

        return action, new_obs

    def process_sim_observation(self, raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy()
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy()
        assert (gripper_state <= 0.04).all(), gripper_state
        gripper_state = 1 - gripper_state / 0.04  # 1 is closed, 0 is open
        if gripper_state > 0.2:
            gripper_state = 1.0
        else:
            gripper_state = 0.0

        eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
        # conver quaternion to euler angles
        eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
        eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
        eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

        images = {}
        images["0"] = raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()
        images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()

        # wrist_img_resolution = (320, 240)
        # wrist_img = np.zeros(
        #     (len(images["2"]), wrist_img_resolution[1], wrist_img_resolution[0], 3)
        # )
        # for i in range(len(images["2"])):
        #     wrist_img[i] = cv2.resize(images["2"][i], wrist_img_resolution)
        # images["2"] = wrist_img

        obs = {
            "robot_state": {
                "joint_positions": joint_state,
                "gripper_position": gripper_state,
                "cartesian_position": eef_pos_euler,
            },
            "image": images,
        }
        return obs

    def postprocess_sim_gripper_action(self, action):
        action[..., -1] = -(action[..., -1] * 2 - 1)
        # action[..., -1] = (action[..., -1] > 0).astype(np.float32) * 2 - 1
        # print(action[..., -1])
        # print((action[..., -1] > 0).sum())
        # if (action[..., -1] > 0).sum() > 4:
        #     action[..., -1] = 1
        # else:
        #     action[..., -1] = -1
        return action

    def run(self):
        # Reset env before iteration starts
        self.model.eval()
        self.reset_sim_env()
        if self.env.control_mode == "pd_ee_pose":
            action = self.env.agent.tcp_pose()
        elif self.env.control_mode == "pd_joint_pos":
            action = self.env.agent.robot.get_qpos()[:, :8]
            action[:, -1] = 1.0
        else:
            raise NotImplementedError

        for _ in range(10):
            prev_obs, rew, terminated, truncated, info = self.env.step(action)
            self.env.render()

        env_success = torch.zeros(self.env.num_envs).to(torch.bool)
        env_states = {
            "object": {
                "manip_obj": self.env.manip_obj.pose.raw_pose.cpu().numpy().copy(),
                "goal_obj": self.env.goal_obj.pose.raw_pose.cpu().numpy().copy(),
            },
            "robot": {
                "qpos": self.env.agent.robot.get_qpos().cpu().numpy().copy(),
                "base_pose": self.env.agent.robot.pose.raw_pose.cpu().numpy().copy(),
            },
            "lighting": self.env.lighting.copy(),
            "camera_pose": self.env.camera_pose.copy(),
            "floor_texture_file": self.env.table_scene.floor_texture_file,
            "table_model_file": self.env.table_scene.table_model_file,
            "success": env_success.cpu().numpy(),
            "eval_steps": [0] * self.env.num_envs,
            "env_elapsed_steps": [0] * self.env.num_envs,
        }

        env_states, env_success = update(
            env_states, env_success, terminated.cpu(), info
        )

        prev_obs = self.process_sim_observation(prev_obs)
        obs = prev_obs
        np.set_printoptions(precision=3, suppress=True)
        if self.cfg.num_views == 2:
            camera_indices = ["0", "2"]
        else:
            camera_indices = ["0"]

        # Check inference
        print("Warming up policy inference")
        with torch.no_grad():
            cond = {}
            cond["state"] = self.process_multistep_state(obs=obs, prev_obs=prev_obs)
            cond["rgb"] = self.process_multistep_img(
                obs=obs,
                camera_indices=camera_indices,
                prev_obs=prev_obs,
                bgr2rgb=False,
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
            naction = samples[:, : self.act_steps]
            prev_obs = obs
        if self.use_delta_actions:
            cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
            action = self.unnormalize_delta_action(naction, cur_state)
            print("*using delta")
        else:
            action = self.unnormalize_action(naction)
        action = self.postprocess_sim_gripper_action(action)
        print("Action:", action)
        print("States:", cond["state"].cpu().numpy())

        # Check images
        for i in camera_indices:
            plt.imshow(
                cond["rgb"][i][0, 0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            plt.axis("off")  # Turn off the axes
            plt.gca().set_position(
                [0, 0, 1, 1]
            )  # Remove white space by adjusting the position of the axes
            plt.savefig(f"image_{i}.png", bbox_inches="tight", pad_inches=0)
            # plt.close()

        # input("Check images aPnd then press anything to continue...")

        # Run
        # cond_states = []
        # images = [[], []]
        # true_cond_states = []
        # true_images = [[], []]
        # actions = []
        # robot_states = [
        #     np.concatenate(
        #         [
        #             obs["robot_state"]["cartesian_position"],
        #             obs["robot_state"]["gripper_position"],
        #         ],
        #         axis=1,
        #     )
        # ]
        # camera_indices = [0, 2]
        # current_gripper = 1.0
        try:
            # for step in range(200, 10000):
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
                        bgr2rgb=False,
                    )
                    # cv2.imwrite(
                    #     f"image_{camera_indices[1]}.png",
                    #     cond["rgb"][camera_indices[1]]
                    #     .squeeze()
                    #     .cpu()
                    #     .numpy()
                    #     .transpose(1, 2, 0)
                    #     .astype(np.uint8),
                    # )
                    # cond["state"] = torch.tensor(
                    #     self.states[self.state_timestep[step]]
                    # )[None, :].to(self.device)
                    # cond["rgb"] = {
                    #     i: torch.tensor(self.images[i][self.state_timestep[step]])[
                    #         None, :
                    #     ][None, :].to(self.device)
                    #     for i in camera_indices
                    # }

                    # cond_states.append(cond["state"].cpu().numpy())
                    # for i, k in enumerate(cond["rgb"].keys()):
                    #     images[i].append(cond["rgb"][k].cpu().numpy())

                    # true_cond_states.append(
                    #     self.states[self.state_timestep[step]][None, :]
                    # )
                    # for i, k in enumerate(cond["rgb"].keys()):
                    #     true_images[i].append(
                    #         self.images[int(k)][self.state_timestep[step]][None, :]
                    #     )

                    pre_obs_end_time = time.time()
                    print(f"Pre-obs time: {pre_obs_end_time - pre_obs_start_time}")

                    # Run forward pass
                    samples = (
                        self.model.sample(cond=cond, deterministic=True)
                        .trajectories.cpu()
                        .numpy()
                    )
                    naction = samples[
                        :, : self.act_steps
                    ]  # (num_envs, act_steps, action_dim)
                    # naction = self.actions[
                    #     self.action_timestep[step] : self.action_timestep[step]
                    #     + self.act_steps
                    # ][None, :]
                    # print("Action chunk:", naction)
                if self.use_delta_actions:
                    cur_state = self.unnormalize_obs(cond["state"].cpu().numpy())
                    action = self.unnormalize_delta_action(naction, cur_state)
                    print("using delta")
                else:
                    action = self.unnormalize_action(naction)
                action = self.postprocess_sim_gripper_action(action)
                # actions.append(action)

                # Debug
                model_inf_end_time = time.time()
                print(f"Model inference time: {model_inf_end_time - pre_obs_end_time}")
                print("Action: ", action)

                # Run action chunk
                for action_step in range(self.act_steps):
                    a = action[:, action_step]
                    prev_obs = obs
                    step_start_time = time.time()

                    # if a[..., -1] != current_gripper:
                    #     print(a[..., -1].item(), current_gripper)
                    #     breakpoint()
                    #     current_gripper = a[..., -1].item()
                    #     step_start_time = time.time()
                    #     for _ in range(50):
                    #         obs, rew, terminated, truncated, info = self.env.step(a)
                    #         env_states, env_success = update(
                    #             env_states,
                    #             env_success,
                    #             terminated.cpu(),
                    #             info,
                    #             eval_step=step,
                    #         )
                    #         self.env.render()
                    #         step_end_time = time.time()
                    #         print(f"Step time: {step_end_time - step_start_time}")
                    #         if a[..., -1].item() == 1.0:
                    #             print("Opening gripper")
                    #         else:
                    #             print("Closing gripper")
                    #     obs = self.process_sim_observation(obs)
                    #     breakpoint()
                    #     break

                    # a = np.concatenate([a, a[..., -1:]], axis=1)
                    # a[..., -2:] = (a[..., -2:] + 1) * 0.02
                    # self.env.agent.robot.set_qpos(a)
                    # obs, info = step_without_action(self.env)
                    # terminated = torch.tensor(0).to(torch.bool)

                    obs, rew, terminated, truncated, info = self.env.step(a)
                    # obs, rew, terminated, truncated, info = self.env.step(a)
                    env_states, env_success = update(
                        env_states, env_success, terminated.cpu(), info, eval_step=step
                    )
                    self.env.render()
                    step_end_time = time.time()
                    print(f"Step time: {step_end_time - step_start_time}")
                    obs = self.process_sim_observation(obs)
                    # robot_states.append(
                    #     np.concatenate(
                    #         [
                    #             obs["robot_state"]["cartesian_position"],
                    #             obs["robot_state"]["gripper_position"],
                    #         ],
                    #         axis=1
                    #     )
                    # )
                    # time.sleep(0.03)
                if env_success:
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        # np.savez_compressed(
        #     "val.npz",
        #     cond_states=np.array(cond_states),
        #     images=np.array(images),
        #     true_cond_states=np.array(true_cond_states),
        #     true_images=np.array(true_images),
        # )

        return env_states
