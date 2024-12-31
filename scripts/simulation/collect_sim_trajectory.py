import logging
import os

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from guided_dc.utils.hdf5_utils import save_dict_to_hdf5
from guided_dc.utils.io_utils import load_hdf5, load_sim_hdf5, stack_videos_horizontally
import omegaconf

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

    images = {}
    images["0"] = raw_obs[0]["sensor_data"]["sensor_0"]["rgb"].cpu().numpy()[0]
    images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()[0]

    return joint_state, gripper_state, images


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


def override_cfg(cfg, override):
    
    def find_and_set_key(d, target_key, value):
        if not isinstance(d, omegaconf.dictconfig.DictConfig):
            return False
        
        for key in d:
            if key == target_key:
                d[key] = value
                return True
            if isinstance(d[key], omegaconf.dictconfig.DictConfig):
                if find_and_set_key(d[key], target_key, value):
                    return True
        return False
    
    for k, v in override.items():
        find_and_set_key(cfg, k, v)

@hydra.main(
    config_path=os.path.join(os.getcwd(), "guided_dc/cfg/simulation"),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg):
    OmegaConf.resolve(cfg)

    real_folder = "tomato_plate_trials-date"

    np.set_printoptions(suppress=True, precision=3)

    base_position = np.array(
        [-0.547, -0.527, -0.143]
    )  # Position of the robot base in the global frame
    base_orientation = np.array([0, 0, np.pi / 4])

    ambient_to_sample = [
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
        [0.8, 0.8, 0.8],
        [0.4, 0.4, 0.4],
    ]

    x_range = np.linspace(-0.1, 0.2, 4)
    y_range = np.linspace(-0.15, 0.15, 3)

    pear_pos_to_sample = []
    bowl_pos_to_sample = []
    apple_pos_to_sample = []

    for x in x_range:
        for y in y_range:
            x = float(x)
            y = float(y)
            pear_pos_to_sample.append([x, y, 0.3])
            bowl_pos_to_sample.append([x, y, 0.1])
            apple_pos_to_sample.append([x, y, 0.15])

    distractors_to_sample = ["pear", "bowl", "apple"]
    floor_texture_files_to_sample = [
        f"{ASSET_DIR}/floor/medium_brown_wood.jpg",
        f"{ASSET_DIR}/floor/dark_brown_wood.jpg",
        f"{ASSET_DIR}/floor/light_brown_wood.jpg",
        f"{ASSET_DIR}/floor/silver_wood.jpg",
    ]

    for traj_idx in range(75):
        try:
            real_traj_dict, _ = load_hdf5(
                file_path=f"{DATA_DIR}/{real_folder}/{traj_idx}.h5",
                action_keys=["joint_position", "gripper_position"],
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
        actions = np.concatenate(
            [
                real_js,
                real_gs[:, None],
            ],
            axis=1,
        )
        actions[:, -1] = (actions[:, -1] > 0.2).astype(np.float32)

        actions[:, -1] = -(actions[:, -1] * 2 - 1)
        # For the gripper position, we round values <0 to -1, and values >0 to 1
        for i in range(len(actions)):
            if actions[i, -1] < 0:
                actions[i, -1] = -1
            else:
                actions[i, -1] = 1
        print(actions[:, -1])

        for trial in range(15):
            distractor_num_this_trial = np.random.randint(1, 3)
            distractor_this_trial = np.random.choice(
                distractors_to_sample, size=distractor_num_this_trial, replace=False
            )
            distractor_dict = []
            for distractor in distractor_this_trial:
                if distractor == "pear":
                    distractor_dict.append(
                        {
                            "type": "custom",
                            "obj_name": "pear",
                            "model_file": f"{ASSET_DIR}/objects/pear.glb",
                            "pos": pear_pos_to_sample[
                                np.random.choice(len(pear_pos_to_sample))
                            ],
                            "rot": [0, 0, 0],
                            "scale": [0.05, 0.05, 0.05],
                        }
                    )
                elif distractor == "bowl":
                    distractor_dict.append(
                        {
                            "type": "custom",
                            "obj_name": "bowl",
                            "model_file": f"{ASSET_DIR}/objects/bowl.glb",
                            "pos": bowl_pos_to_sample[
                                np.random.choice(len(bowl_pos_to_sample))
                            ],
                            "rot": [0, 0, 0],
                            "scale": [0.01, 0.01, 0.01],
                        }
                    )
                elif distractor == "apple":
                    distractor_dict.append(
                        {
                            "type": "custom",
                            "obj_name": "apple",
                            "model_file": f"{ASSET_DIR}/objects/apple.glb",
                            "pos": apple_pos_to_sample[
                                np.random.choice(len(apple_pos_to_sample))
                            ],
                            "rot": [0, 0, 0],
                            "scale": [0.01, 0.01, 0.01],
                        }
                    )

            ambient_this_trial = ambient_to_sample[
                np.random.choice(len(ambient_to_sample))
            ]
            floor_texture_file_this_trial = floor_texture_files_to_sample[
                np.random.choice(len(floor_texture_files_to_sample))
            ]

            override = OmegaConf.create(
                {
                    "distractor": distractor_dict,
                    "floor_texture_file": floor_texture_file_this_trial,
                    "ambient": ambient_this_trial,
                }
            )
            override_cfg(cfg, override)
            print(cfg.env.scene_builder.distractor)

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
                env.reset()
                env_start_js = env.agent.robot.get_qpos().cpu().numpy().squeeze()[:7]
                if np.linalg.norm(traj_start_js - env_start_js) > 0.1:
                    js_to_set = np.concatenate([traj_start_js, [0.04, 0.04]])
                    env.agent.robot.set_qpos(js_to_set)
                    step(env)

                # 1. Save real action trajectory
                jss = []
                gss = []
                wrist_imgs = []
                side_imgs = []
                acts = []

                success = False
                for timestep, action in enumerate(actions):
                    if timestep == pick_obj_timestep:
                        tcp_pos = env.agent.tcp.pose.p.cpu().numpy().squeeze()
                        pick_offset = tcp_pos - global_pick_pos
                    if timestep == place_obj_timestep:
                        tcp_pos = env.agent.tcp.pose.p.cpu().numpy().squeeze()
                        place_offset = tcp_pos - global_place_pos

                    obs, rew, terminated, truncated, info = env.step(action)
                    success = success or terminated
                    print(action)
                    js, gs, img = process_sim_observation(obs)
                    jss.append(js)
                    gss.append(gs)
                    side_imgs.append(img["0"])
                    wrist_imgs.append(img["2"])
                    acts.append(action)

                # save_array_to_video("temp.mp4", wrist_imgs, fps=30, brg2rgb=False)

            pick_obj_pos = cfg.env.manip_obj.pos
            place_obj_pos = cfg.env.goal_obj.pos

            jss = np.array(jss)
            gss = np.array(gss)
            wrist_imgs = np.array(wrist_imgs).astype(np.uint8)
            side_imgs = np.array(side_imgs).astype(np.uint8)
            acts = np.array(acts)

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
                    },
                },
                "action": {
                    "joint_position": acts[:, :-1],
                    "gripper_position": acts[:, -1],
                },
                "pick_obj_pos": pick_obj_pos,
                "place_obj_pos": place_obj_pos,
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

            os.makedirs(f"{DATA_DIR}/sim/{real_folder}", exist_ok=True)
            # if os.path.exists(f"{DATA_DIR}/sim/{traj_idx}_sim.h5"):
            #     os.path.remove(f"{DATA_DIR}/sim/{traj_idx}_sim.h5")
            if not success:
                save_dict_to_hdf5(
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim_failed.h5",
                    data_dict,
                )
                a = load_sim_hdf5(
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim_failed.h5",
                    load_o=False,
                )[0]
            else:
                save_dict_to_hdf5(
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim.h5", data_dict
                )
                a = load_sim_hdf5(
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim.h5",
                    load_o=False,
                )[0]

            videos_1 = a["observation/image/0"]
            videos_2 = a["observation/image/2"]
            if not success:
                stack_videos_horizontally(
                    videos_1,
                    videos_2,
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim_failed.mp4",
                )
            else:
                stack_videos_horizontally(
                    videos_1,
                    videos_2,
                    f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_{trial}_sim.mp4",
                )
            # video_1 = a["observation_o/image/0"]
            # video_2 = a["observation_o/image/2"]
            # stack_videos_horizontally(
            #     video_1, video_2, f"{DATA_DIR}/sim/{real_folder}/{traj_idx}_sim_o.mp4"
            # )

    env.close()


if __name__ == "__main__":
    main()
