import logging
import os

import cv2
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "guided_dc/cfg/simulation"
    ),  # possibly overwritten by --config-path
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg):
    OmegaConf.resolve(cfg)

    if not cfg.quiet:
        log.info(f"Loaded configuration: \n{OmegaConf.to_yaml(cfg)}")

    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")

    env = gym.make(cfg.env.env_id, cfg=cfg.env)

    output_dir = cfg.record.output_dir
    render_mode = cfg.env.render_mode

    cfg.record.save_video = False

    if output_dir and render_mode != "human":
        log.info(f"Recording environment episodes to: {output_dir}")
        env = RecordEpisode(
            env,
            max_steps_per_video=env._max_episode_steps,
            **cfg.record,
        )

    # Log environment details if verbose output is enabled
    if not cfg.quiet:
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
        log.info(f"Control mode: {env.unwrapped.control_mode}")
        log.info(f"Reward mode: {env.unwrapped.reward_mode}")

    if (not render_mode == "human") and (cfg.record.save_trajectory):
        env._json_data["env_info"]["env_kwargs"] = OmegaConf.create(
            env._json_data["env_info"]["env_kwargs"]
        )
        env._json_data["env_info"]["env_kwargs"] = OmegaConf.to_container(
            env._json_data["env_info"]["env_kwargs"]
        )

    def process_sim_observation(raw_obs):
        if isinstance(raw_obs, dict):
            raw_obs = [raw_obs]
        images = {}
        images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()
        wrist_img_resolution = (320, 240)
        wrist_img = np.zeros(
            (len(images["2"]), wrist_img_resolution[1], wrist_img_resolution[0], 3)
        )
        for i in range(len(images["2"])):
            wrist_img[i] = cv2.resize(images["2"][i], wrist_img_resolution)
        images["2"] = wrist_img
        return wrist_img[0]

    # # Read the ground truth image from disk
    # gt_wrist = cv2.imread("gt_wrist.png")
    # # Resize the ground truth image to (240, 320)
    # gt_wrist_resized = cv2.resize(gt_wrist, (320, 240))
    # # Convert BGR to RGB for proper visualization
    # gt_wrist_resized = cv2.cvtColor(gt_wrist_resized, cv2.COLOR_BGR2RGB)

    # def visualize_obs(obs, q=None, idx=0):
    #     wrist_img = process_sim_observation(obs)
    #     # plt.imshow(wrist_img.astype(np.uint8))
    #     # plt.savefig("image_2.png")

    #     # Plot the two images side by side
    #     fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    #     fig.subplots_adjust(wspace=0.05, hspace=0)  # Reduce space between plots

    #     # Display the wrist image from the simulation
    #     axes[0].imshow(wrist_img.astype(np.uint8))
    #     # axes[0].set_title("Simulated Image")
    #     axes[0].axis("off")

    #     # Display the resized ground truth image
    #     axes[1].imshow(gt_wrist_resized)
    #     # axes[1].set_title("Ground Truth Image")
    #     axes[1].axis("off")

    #     if q is not None:
    #         formatted_q = ", ".join(f"{value:.3f}" for value in q)
    #         plt.suptitle(f"Quaternion: [{formatted_q}]")

    #     # Save the combined figure
    #     plt.savefig(f"./comparison/{idx}.png")

    obs, info = env.reset()

    if env.control_mode == "pd_ee_pose":
        action = env.agent.tcp_pose()
    elif env.control_mode == "pd_joint_pos":
        action = env.agent.robot.get_qpos()[:, :8]
        action[:, -1] = 1.0
    else:
        raise NotImplementedError

    from mani_skill.utils import sapien_utils

    from guided_dc.utils.pose_utils import quaternion_to_euler_xyz  # noqa: F401

    # Define initial quaternion and search range
    # initial_q = [-0.8052, -0.1164, -0.4494, 0.3689]

    # initial_q = [-0.798, -0.191, -0.396, 0.413]

    # initial_q = [
    #     0.814,
    #     -0.094,
    #     0.23055,
    #     -0.5237,
    # ]
    initial_p = [-0.3662, -0.1873, 0.3187]
    pose = sapien_utils.look_at(eye=initial_p, target=[-0.20, -0.37, 0])
    # initial_T = np.eye(4)
    # Transform a wxyz quaternion to a rotation matrix
    # from wxyz to xyzw
    # initial_q = [initial_q[1], initial_q[2], initial_q[3], initial_q[0]]
    # initial_T[:3, :3] = R.from_quat(initial_q).as_matrix()
    # initial_T[:3, 3] = initial_p

    # scale = 0.0

    # delta_R = np.array(
    #     [
    #         [
    #             [
    #                 [
    #                     [0.98609038, 0.01945441, -0.16506754],
    #                     [0.00117833, 0.99228314, 0.123987],
    #                     [0.16620583, -0.1224569, 0.97845793],
    #                 ]
    #             ]
    #         ]
    #     ]
    # )
    # delta_p = np.array([-0.9609755, -0.07149666, -0.26723456]) * scale
    # # delta_p = np.array([0.0, 0.0, 0.0])
    # delta_T = np.eye(4)
    # delta_T[:3, :3] = delta_R
    # delta_T[:3, 3] = delta_p

    # T_b = np.dot(initial_T, delta_T)
    # quat_b = R.from_matrix(T_b[:3, :3]).as_quat()
    # # from xyzw to wxyz
    # quat_b = [quat_b[3], *quat_b[:3]]

    # p_b = T_b[:3, 3]

    # env.wrist_mount.set_pose(
    #     Pose.create_from_pq(
    #         # p=p_b,
    #         # q=quat_b,
    #         p=initial_p,
    #         q=initial_q,
    #     )
    # )
    env.wrist_mount.set_pose(pose)
    panda_hand_pose = env.agent.robot.links_map["panda_hand"].pose
    wrist_mount_pose = env.wrist_mount.pose
    relative_pose = panda_hand_pose.inv() * wrist_mount_pose
    print("Relative pose to the panda hand:\n", relative_pose)

    # Reset the environment and visualize the observation
    obs, _ = env.reset()
    # for _ in range(1):
    #     env.step(action)
    #     env.render()
    # cv2.imwrite('rgba.png',obs['sensor_data']['hand_camera']['rgb'].squeeze().cpu().numpy())
    # depth_image = obs['sensor_data']['hand_camera']['depth'].cpu().numpy().squeeze().astype(np.uint8)
    # color_mapped_depth = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    # cv2.imshow('Color-mapped Depth Map', color_mapped_depth)
    # # cv2.imshow('Depth.png', depth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # breakpoint()
    wrist_img = process_sim_observation(obs)
    plt.imshow(wrist_img.astype(np.uint8))  # Use cmap='gray' if the image is grayscale
    plt.axis("off")  # Turn off the axes
    plt.gca().set_position(
        [0, 0, 1, 1]
    )  # Remove white space by adjusting the position of the axes

    # Save the image
    plt.savefig("image_2_previous.png", bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the plot to avoid displaying it
    # visualize_obs(obs, initial_q, 0)

    # initial_euler = quaternion_to_euler_xyz(initial_q).squeeze()

    # search_range = 0.2

    # # # Define ranges for the orientation components (euler) around the initial values
    # q1_range = np.linspace(initial_q[0] - search_range, initial_q[0] + search_range, 5)
    # q2_range = np.linspace(initial_q[1] - search_range, initial_q[1] + search_range, 5)
    # q3_range = np.linspace(initial_q[2] - search_range, initial_q[2] + search_range, 5)
    # q4_range = np.linspace(initial_q[3] - search_range, initial_q[3] + search_range, 5)
    # # # roll_range = np.linspace(
    # # #     initial_euler[0] - search_range, initial_euler[0] + search_range, 10
    # # # )
    # # # pitch_range = np.linspace(
    # # #     initial_euler[1] - search_range, initial_euler[1] + search_range, 10
    # # # )
    # # # yaw_range = np.linspace(
    # # #     initial_euler[2] - search_range, initial_euler[2] + search_range, 10
    # # # )

    # idx = 0

    # # # for roll in roll_range:
    # # #     for pitch in pitch_range:
    # # #         for yaw in yaw_range:
    # # #             q = euler_to_quaternion([roll, pitch, yaw])
    # # #             print(q)
    # # #             # Set the camera pose with the current orientation q
    # # #             env.wrist_mount.set_pose(
    # # #                 Pose.create_from_pq(
    # # #                     p=[-0.3762, -0.2073, 0.3237],
    # # #                     q=q,
    # # #                 )
    # # #             )

    # # #             # Reset the environment and visualize the observation
    # # #             obs, _ = env.reset()
    # # #             visualize_obs(obs, q, idx)
    # # #             idx += 1

    # # Iterate through all combinations of q components
    # for q1 in q1_range:
    #     for q2 in q2_range:
    #         for q3 in q3_range:
    #             for q4 in q4_range:
    #                 # Normalize quaternion to ensure it's valid
    #                 norm = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
    #                 if norm == 0:
    #                     continue  # Skip invalid quaternions
    #                 q = [q1 / norm, q2 / norm, q3 / norm, q4 / norm]
    #                 print(q)
    #                 # Set the camera pose with the current orientation q
    #                 env.wrist_mount.set_pose(
    #                     Pose.create_from_pq(
    #                         p=[-0.3762, -0.2073, 0.3237],
    #                         q=q,
    #                     )
    #                 )

    #                 # Reset the environment and visualize the observation
    #                 obs, _ = env.reset()
    #                 visualize_obs(obs, q, idx)
    #                 idx += 1

    # # # Read all images from ./comparison and save them as a video
    # img_array = []
    # for i in range(0, idx):
    #     img = cv2.imread(f"./comparison/{i}.png")
    #     height, width, layers = img.shape
    #     size = (width, height)
    #     img_array.append(img)

    # out = cv2.VideoWriter("comparison.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])

    # out.release()

    # if render_mode is not None:
    #     viewer = env.render()
    #     if isinstance(viewer, sapien.utils.Viewer):
    #         viewer.paused = cfg.pause

    # for _ in range(1):
    #     prev_obs, _, _, _, _ = env.step(action)
    #     if render_mode is not None:
    #         env.render()

    env.close()

    # video_name = f"./videos/output_{cfg.env.env_id}.mp4"

    # if output_dir and render_mode != "human":
    #     print(f"Saving video to {output_dir}")
    #     merge_rgb_array_videos(output_dir, video_name)


if __name__ == "__main__":
    main()
