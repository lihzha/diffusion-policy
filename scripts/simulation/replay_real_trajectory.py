import logging
import os

import gymnasium as gym
import hydra
import numpy as np
import sapien
from omegaconf import OmegaConf

from guided_dc.maniskill.mani_skill.utils.wrappers import RecordEpisode
from guided_dc.utils.io_utils import load_hdf5

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

    real_traj_dict, _ = load_hdf5(
        file_path="data/4.h5",
        action_keys=["joint_position", "gripper_position"],
        observation_keys=["joint_positions", "gripper_position"],
        load_image=True,
    )
    actions = np.concatenate(
        [
            real_traj_dict["action/joint_position"],
            real_traj_dict["action/gripper_position"][:, None],
        ],
        axis=1,
    )

    actions[:, -1] = -(actions[:, -1] * 2 - 1)
    # For the gripper position, we round values <0 to -1, and values >0 to 1
    for i in range(len(actions)):
        if actions[i, -1] < 0:
            actions[i, -1] = -1
        else:
            actions[i, -1] = 1
    print(actions[:, -1])

    env = gym.make(cfg.env.env_id, cfg=cfg.env)

    output_dir = cfg.record.output_dir
    render_mode = cfg.env.render_mode
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
        joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy().squeeze(0)
        gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy().squeeze(0)
        # assert (gripper_state <= 0.04).all(), gripper_state
        # gripper_state = 1 - gripper_state / 0.04

        images = {}
        images["0"] = raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()[0]
        images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()[0]

        return joint_state, gripper_state, images

    env.reset()

    if render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = cfg.pause

    jss = []
    gss = []
    wrist_imgs = []
    side_imgs = []

    def step():
        """
        Take a step through the environment with an action. Actions are automatically clipped to the action space.

        If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
        """
        info = env.get_info()
        obs = env.get_obs(info)
        return (
            obs,
            info,
        )

    real_js = real_traj_dict["observation/robot_state/joint_positions"]
    real_gs = real_traj_dict["observation/robot_state/gripper_position"]
    real_gs = 0.04 - real_gs * 0.04

    obs = np.concatenate([real_js, real_gs[:, None], real_gs[:, None]], axis=1)

    # real_actions = np.concatenate([real_js, real_gs[:, None]], axis=1)

    # for action in actions:
    for action in obs:
        # for i in range(10):
        # obs, _, _, _, _ = env.step(action)
        # action = np.concatenate([action, [action[-1]]])
        env.agent.robot.set_qpos(action)
        obs, info = step()
        print(action)
        js, gs, img = process_sim_observation(obs)
        jss.append(js)
        gss.append(gs)
        side_imgs.append(img["0"])
        wrist_imgs.append(img["2"])
        if render_mode is not None:
            env.render()

    real_js = real_traj_dict["observation/robot_state/joint_positions"]
    real_gs = real_traj_dict["observation/robot_state/gripper_position"]

    jss = np.array(jss)
    gss = np.array(gss)

    # Use cv2 to save wrist imgs as video
    wrist_imgs = np.array(wrist_imgs).astype(np.uint8)
    side_imgs = np.array(side_imgs).astype(np.uint8)

    np.save("side_imgs_rt.npy", side_imgs)
    np.save("wrist_imgs_rt.npy", wrist_imgs)
    np.save("jss_rt.npy", jss)
    np.save("gss_rt.npy", gss)

    # for i in range(len(real_traj_dict["observation/image/3"])):
    #     real_traj_dict["observation/image/3"][i] = cv2.cvtColor(
    #         real_traj_dict["observation/image/3"][i], cv2.COLOR_BGR2RGB
    #     )
    # stack_videos_horizontally(
    #     wrist_imgs, real_traj_dict["observation/image/3"], "wrist_imgs.mp4", fps=30
    # )

    # Plot real and simulated joint positions in seperate subplots
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))

    for i in range(7):
        axs[i % 2, i // 2].plot(real_js[:, i], label="Real States")
        axs[i % 2, i // 2].plot(jss[:, i], label="Simulated States")
        axs[i % 2, i // 2].plot(actions[:, i], label="Real Actions")
        axs[i % 2, i // 2].set_title(f"Joint {i}")
        axs[i % 2, i // 2].legend()

    gss = 1 - gss / 0.04
    axs[1, 3].plot(real_gs, label="Real")
    axs[1, 3].plot(gss, label="Simulated")
    axs[1, 3].set_title("Gripper position")
    axs[1, 3].legend()

    plt.savefig("real_vs_simulated.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    env.close()

    # video_name = f"output_{cfg.env.env_id}"

    # if output_dir and render_mode != "human":
    #     print(f"Saving video to {output_dir}")
    #     merge_rgb_array_videos(output_dir, name=video_name)


if __name__ == "__main__":
    main()
