import logging
import os

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from guided_dc.utils.hdf5_utils import save_dict_to_hdf5
from guided_dc.utils.io_utils import load_hdf5, stack_videos_horizontally
from guided_dc.utils.pose_utils import quaternion_to_euler_xyz

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)

log = logging.getLogger(__name__)
DATA_DIR = os.environ["GDC_DATA_DIR"]
ASSET_DIR = os.environ["GDC_ASSETS_DIR"]


def find_pick_and_place_times(gripper_position):
    pick_start = -1
    place_start = -1

    # Find the start of the last subsequence of 1
    for i in range(len(gripper_position) - 1, -1, -1):
        if gripper_position[i] == 1:
            pick_start = i
        elif pick_start != -1:
            # Found the start of the last subsequence of 1
            break

    # Find the start of the immediate subsequence of 0 after pick_start
    if pick_start != -1:
        for i in range(pick_start + 1, len(gripper_position)):
            if gripper_position[i] == 0:
                place_start = i
                break

    return pick_start, place_start


def transform_to_global(relative_pos, base_position, base_orientation):
    """
    Transforms a position from the robot base's frame to the global frame.

    :param relative_pos: np.array, shape (3,), position relative to the robot base
    :param base_position: np.array, shape (3,), position of the robot base in the global frame
    :param base_orientation: np.array, shape (3,)
    :return: np.array, shape (3,), position in the global frame
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_euler("xyz", base_orientation)  # Quaternion format: [x, y, z, w]
    rotation_matrix = rotation.as_matrix()

    # Transform the position
    global_pos = rotation_matrix @ relative_pos + base_position
    return global_pos


def process_sim_observation(raw_obs):
    if isinstance(raw_obs, dict):
        raw_obs = [raw_obs]
    joint_state = raw_obs[0]["agent"]["qpos"][:, :7].cpu().numpy().squeeze(0)
    gripper_state = raw_obs[0]["agent"]["qpos"][:, 7:8].cpu().numpy().squeeze(0)
    # assert (gripper_state <= 0.04).all(), gripper_state
    # gripper_state = 1 - gripper_state / 0.04

    eef_pos_quat = raw_obs[0]["extra"]["tcp_pose"].cpu().numpy()
    # conver quaternion to euler angles
    eef_pos_euler = np.zeros((eef_pos_quat.shape[0], 6))
    eef_pos_euler[:, :3] = eef_pos_quat[:, :3]
    eef_pos_euler[:, 3:] = quaternion_to_euler_xyz(eef_pos_quat[:, 3:])

    images = {}
    images["0"] = raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()[0]
    images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()[0]

    return joint_state, gripper_state, images, eef_pos_euler


def step(env):
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


@hydra.main(
    config_path=os.path.join(os.getcwd(), "guided_dc/cfg/simulation"),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg):
    OmegaConf.resolve(cfg)

    real_folders = ["tomato_plate_trials-date", "tomato_plate_dec1"]
    traj_nums = [75, 50]

    np.set_printoptions(suppress=True, precision=3)

    base_position = np.array(
        [-0.547, -0.527, -0.143]
    )  # Position of the robot base in the global frame
    base_orientation = np.array([0, 0, np.pi / 4])

    for real_folder, traj_num in zip(real_folders, traj_nums):
        pick_obj_poses = []
        place_obj_poses = []

        for traj_idx in range(traj_num):
            try:
                real_traj_dict, _ = load_hdf5(
                    file_path=f"{DATA_DIR}/{real_folder}/{traj_idx}.h5",
                    action_keys=[
                        "joint_position",
                        "gripper_position",
                        "cartesian_position",
                    ],
                    observation_keys=[
                        "joint_positions",
                        "gripper_position",
                        "cartesian_position",
                    ],
                    load_image=True,
                )
            except FileNotFoundError:
                continue
            # actions = np.concatenate(
            #     [
            #         real_traj_dict["action/joint_position"],
            #         real_traj_dict["action/gripper_position"][:, None],
            #     ],
            #     axis=1,
            # )

            pick_obj_timestep, place_obj_timestep = find_pick_and_place_times(
                real_traj_dict["action/gripper_position"]
            )
            relative_pick_pos = np.array(
                real_traj_dict["observation/robot_state/cartesian_position"][
                    pick_obj_timestep
                ][:3]
            )
            global_pick_pos = transform_to_global(
                relative_pick_pos, base_position, base_orientation
            )

            relative_place_pos = np.array(
                real_traj_dict["observation/robot_state/cartesian_position"][
                    place_obj_timestep
                ][:3]
            )
            global_place_pos = transform_to_global(
                relative_place_pos, base_position, base_orientation
            )

            real_js = real_traj_dict["observation/robot_state/joint_positions"]
            real_gs = real_traj_dict["observation/robot_state/gripper_position"]
            real_gs = (real_gs > 0.2).astype(np.float32)
            real_gs = -real_gs * 2 + 1
            # For the gripper position, we round values <0 to -1, and values >0 to 1
            for i in range(len(real_gs)):
                if real_gs[i] < 0:
                    real_gs[i] = -1
                else:
                    real_gs[i] = 1

            actions = np.concatenate(
                [
                    real_js,
                    real_gs[:, None],
                ],
                axis=1,
            )

            eef_actions = real_traj_dict["action/cartesian_position"]

            print(actions[:, -1])

            pick_offset = np.zeros(3)
            place_offset = np.zeros(3)

            traj_start_js = actions[0, :-1]

            for _ in range(2):
                cfg.env.manip_obj.pos[:2] = [
                    float(v) for v in global_pick_pos[:2] + pick_offset[:2]
                ]
                cfg.env.goal_obj.pos[:2] = [
                    float(v) for v in global_place_pos[:2] + place_offset[:2]
                ]

                cfg.env.goal_obj.rot[1] = np.random.uniform(-np.pi, np.pi)

                # rand_range = 0.02
                # cfg.env.manip_obj.pos[0] += np.random.uniform(-rand_range, rand_range)
                # cfg.env.manip_obj.pos[1] += np.random.uniform(-rand_range, rand_range)
                # cfg.env.goal_obj.pos[0] += np.random.uniform(-rand_range, rand_range)
                # cfg.env.goal_obj.pos[1] += np.random.uniform(-rand_range, rand_range)

                env = gym.make(cfg.env.env_id, cfg=cfg.env)
                obs, _ = env.reset()
                env_start_js = env.agent.robot.get_qpos().cpu().numpy().squeeze()[:7]
                if np.linalg.norm(traj_start_js - env_start_js) > 0.1:
                    js_to_set = np.concatenate([traj_start_js, [0.04, 0.04]])
                    env.agent.robot.set_qpos(js_to_set)
                    obs, _ = step(env)

                # 1. Save real action trajectory
                jss = []
                gss = []
                wrist_imgs = []
                side_imgs = []
                acts = []
                eef_poses = []

                success = False
                for timestep, action in enumerate(actions):
                    js, gs, img, eef_pose = process_sim_observation(obs)
                    jss.append(js)
                    gss.append(gs)
                    side_imgs.append(img["0"])
                    wrist_imgs.append(img["2"])
                    acts.append(action)
                    eef_poses.append(eef_pose)

                    print(action)
                    if timestep == pick_obj_timestep:
                        tcp_pos = env.agent.tcp.pose.p.cpu().numpy().squeeze()
                        pick_offset = tcp_pos - global_pick_pos
                    if timestep == place_obj_timestep:
                        tcp_pos = env.agent.tcp.pose.p.cpu().numpy().squeeze()
                        place_offset = tcp_pos - global_place_pos

                    obs, rew, terminated, truncated, info = env.step(action)
                    success = success or terminated

                    # save_array_to_video("temp.mp4", wrist_imgs, fps=30, brg2rgb=False)

            pick_obj_pos = cfg.env.manip_obj.pos
            place_obj_pos = cfg.env.goal_obj.pos
            pick_obj_rot = cfg.env.manip_obj.rot
            place_obj_rot = cfg.env.goal_obj.rot

            pick_obj_poses.append([*pick_obj_pos, *pick_obj_rot])
            place_obj_poses.append([*place_obj_pos, *place_obj_rot])

            jss = np.array(jss)
            gss = np.array(gss)
            wrist_imgs = np.array(wrist_imgs).astype(np.uint8)
            side_imgs = np.array(side_imgs).astype(np.uint8)
            acts = np.array(acts)
            eef_poses = np.array(eef_poses)

            # 2. Save real observation trajectory
            # env.reset()
            # jss_o = []
            # gss_o = []
            # wrist_imgs_o = []
            # side_imgs_o = []
            # acts_o = []

            # real_js = real_traj_dict["observation/robot_state/joint_positions"]
            # real_gs = real_traj_dict["observation/robot_state/gripper_position"]
            # real_gs = 0.04 - real_gs * 0.04
            # obs = np.concatenate([real_js, real_gs[:, None], real_gs[:, None]], axis=1)

            # for action in obs:
            #     env.agent.robot.set_qpos(action)
            #     obs, info = step(env)
            #     print(action)
            #     js, gs, img = process_sim_observation(obs)
            #     jss_o.append(js)
            #     gss_o.append(gs)
            #     side_imgs_o.append(img["0"])
            #     wrist_imgs_o.append(img["2"])
            #     acts_o.append(action)

            # jss_o = np.array(jss_o)
            # gss_o = np.array(gss_o)
            # wrist_imgs_o = np.array(wrist_imgs_o).astype(np.uint8)
            # side_imgs_o = np.array(side_imgs_o).astype(np.uint8)
            # acts_o = np.array(acts_o)

            # 3. Save to hdf5
            data_dict = {
                "observation": {
                    "image": {
                        "0": side_imgs,
                        "2": wrist_imgs,
                    },
                    "robot_state": {
                        "joint_positions": jss,
                        "gripper_position": gss.squeeze(),
                        "cartesian_position": eef_poses,
                    },
                },
                "action": {
                    "joint_position": acts[:, :-1],
                    "gripper_position": acts[:, -1],
                    "cartesin_position": eef_actions,
                },
                "pick_obj_pos": pick_obj_pos,
                "place_obj_pos": place_obj_pos,
                "pick_obj_rot": pick_obj_rot,
                "place_obj_rot": place_obj_rot,
                "pick_obj_timestep": pick_obj_timestep,
                "place_obj_timestep": place_obj_timestep,
                # "observation_o": {
                #     "image": {
                #         "0": side_imgs_o,
                #         "2": wrist_imgs_o,
                #     },
                #     "robot_state": {
                #         "joint_positions": jss_o,
                #         "gripper_position": gss_o.squeeze(),
                #     },
                # },
                # "action_o": {
                #     "joint_position": acts_o[:, :-1],
                #     "gripper_position": acts_o[:, -1],
                # },
            }
            # Add override to the data_dict
            # data_dict["override"] = OmegaConf.to_container(override)

            traj_folder = f"{DATA_DIR}/sim_with_eef/{real_folder}"

            os.makedirs(traj_folder, exist_ok=True)
            # if os.path.exists(f"{DATA_DIR}/sim/{traj_idx}_sim.h5"):
            #     os.path.remove(f"{DATA_DIR}/sim/{traj_idx}_sim.h5")
            if not success:
                save_dict_to_hdf5(
                    traj_folder + f"/{traj_idx}_sim_failed.h5",
                    data_dict,
                )
                stack_videos_horizontally(
                    data_dict["observation"]["image"]["0"],
                    data_dict["observation"]["image"]["2"],
                    traj_folder + f"/{traj_idx}_sim_failed.mp4",
                )
            else:
                save_dict_to_hdf5(traj_folder + f"/{traj_idx}_sim.h5", data_dict)
                stack_videos_horizontally(
                    data_dict["observation"]["image"]["0"],
                    data_dict["observation"]["image"]["2"],
                    traj_folder + f"/{traj_idx}_sim.mp4",
                )
            # video_1 = a["observation_o/image/0"]
            # video_2 = a["observation_o/image/2"]
            # stack_videos_horizontally(
            #     video_1, video_2, f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_sim_o.mp4"
            # )
        np.save(traj_folder + "/pick_obj_poses.npy", pick_obj_poses)
        np.save(traj_folder + "/place_obj_poses.npy", place_obj_poses)

    env.close()


if __name__ == "__main__":
    main()
