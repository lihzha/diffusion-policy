"""
Script for processing raw teleop data for policy training.

Create a new folder and then save dataset and normalization values in the folder. Also save the config in txt.

"""

import os
import time
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

from guided_dc.utils.io_utils import load_hdf5, load_sim_hdf5_for_training


def resize_image(args):
    img, img_resolution = args
    return cv2.resize(img, (img_resolution[1], img_resolution[0]))  # (W, H)


def resize_images_multiprocessing(raw_img, img_resolution, num_thread=10):
    args = [(raw_img[i], img_resolution) for i in range(raw_img.shape[0])]

    # Use Pool for multiprocessing
    with Pool(processes=num_thread) as pool:
        resized_images = pool.map(resize_image, args)

    resized_img = np.array(resized_images, dtype=np.uint8)
    return resized_img


def process_real_dataset(
    input_paths,
    output_parent_dir,
    num_traj=None,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    horizon_steps=16,
    img_resolution=(96, 96),
    camera_indices=["0", "2"],
    num_thread=10,
    skip_image=False,
    keep_bgr=False,
    use_obs_as_action=False,
    obs_gripper_threshold=0.2,
    additional_name="",
):
    save_image = not skip_image
    bgr2rgb = not keep_bgr

    # concatenate all paths
    traj_paths = []
    for path in input_paths:
        traj_paths += [
            os.path.join(path, traj_name)
            for traj_name in os.listdir(path)
            if traj_name.endswith(".h5")
        ]
    num_traj_available = len(traj_paths)

    # process first trajectories
    if num_traj is not None:
        traj_paths = traj_paths[:num_traj]
    else:
        num_traj = num_traj_available
    print(
        f"Processing {num_traj}/{num_traj_available} trajectories with {cpu_count()} cpu threads..."
    )

    # initialize output dictionary
    output = {
        "traj_lengths": [],
        "actions": [],
        "states": [],
        "images": {index: [] for index in camera_indices},
    }
    state_action_diff_mins = []
    state_action_diff_maxs = []
    for traj_path in tqdm(traj_paths):
        # load trajectory from h5
        s1 = time.time()
        traj, camera_indices_raw = load_hdf5(
            traj_path,
            action_keys=action_keys,
            observation_keys=observation_keys,
            load_image=save_image,
        )
        print("Time to load h5:", time.time() - s1)

        # skip idle actions (skip_action == True)
        keep_idx = ~traj["observation/timestamp/skip_action"]
        traj_length = len(keep_idx)
        output["traj_lengths"].append(traj_length)
        print("Path:", traj_path, "Length:", traj_length)

        # set gripper position to binary: 0 for open, 1 for closed
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = (
                traj["action/gripper_position"] > 0.5
            ).astype(np.float32)
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = (
                traj["observation/robot_state/gripper_position"] > obs_gripper_threshold
            ).astype(np.float32)

        # set roll to always be positive
        if "cartesian_position" in action_keys:
            traj["action/cartesian_position"][:, 3] = np.abs(
                traj["action/cartesian_position"][:, 3]
            )
        if "cartesian_position" in observation_keys:
            traj["observation/robot_state/cartesian_position"][:, 3] = np.abs(
                traj["observation/robot_state/cartesian_position"][:, 3]
            )

        # add dimension to gripper position
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = traj["action/gripper_position"][:, None]
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = traj[
                "observation/robot_state/gripper_position"
            ][:, None]

        # get the maximum difference between the starting state and each action of the chunk at each timestep
        if "joint_positions" in observation_keys and "joint_position" in action_keys:
            state = traj["observation/robot_state/joint_positions"][keep_idx]
            action = traj["action/joint_position"][keep_idx]
        elif (
            "cartesian_position" in observation_keys
            and "cartesian_position" in action_keys
        ):
            state = traj["observation/robot_state/cartesian_position"][keep_idx]
            action = traj["action/cartesian_position"][keep_idx]
        else:
            raise NotImplementedError(
                "For getting the state-action difference, need consistent keys"
            )
        diffs = np.empty((0, action.shape[1]))
        for step in range(horizon_steps // 4, horizon_steps):  # skip early steps
            diff = action[step:] - state[:-step]
            diffs = np.concatenate([diffs, diff], axis=0)
        state_action_diff_mins.append(np.min(diffs, axis=0))
        state_action_diff_maxs.append(np.max(diffs, axis=0))
        if np.isnan(np.sum(diffs)) > 0 or np.isnan(np.sum(diffs)) > 0:
            raise ValueError("NaN in state-action difference")

        # concatenate states and actions
        if use_obs_as_action:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][
                        :-1
                    ]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][1:]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            output["traj_lengths"][-1] = output["traj_lengths"][-1] - 1
        else:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"action/{action_keys[i]}"][keep_idx]
                    for i in range(len(action_keys))
                ],
                axis=1,
            )
        assert len(states) == output["traj_lengths"][-1]
        assert len(actions) == output["traj_lengths"][-1]
        output["states"].append(states)
        output["actions"].append(actions)

        # add images
        if save_image:
            # verify camera indices
            camera_indices_chosen = [camera_indices_raw[idx] for idx in camera_indices]
            print(f"Using raw camera indices: {camera_indices_chosen}")
            for raw_idx, idx in zip(camera_indices_chosen, camera_indices):
                raw_img = traj[f"observation/image/{raw_idx}"][keep_idx]  # (T, H, W, C)
                assert raw_img.dtype == np.uint8

                # resize with multiprocessing
                s1 = time.time()
                resized_img = resize_images_multiprocessing(
                    raw_img,
                    img_resolution,
                    num_thread,
                )
                print("Time to resize images:", time.time() - s1)

                # Transpose to (T, C, H, W)
                resized_img = resized_img.transpose(0, 3, 1, 2)

                # Change BGR (cv2 default) to RGB
                if bgr2rgb:
                    resized_img = resized_img[:, [2, 1, 0]]

                if use_obs_as_action:
                    resized_img = resized_img[:-1]

                # save
                assert len(resized_img) == output["traj_lengths"][-1]
                output["images"][idx].append(resized_img)

    # Convert to numpy arrays
    output["traj_lengths"] = np.array(output["traj_lengths"])
    output["actions"] = np.concatenate(output["actions"], axis=0)
    output["states"] = np.concatenate(output["states"], axis=0)
    for idx in camera_indices:
        output["images"][idx] = np.concatenate(output["images"][idx], axis=0)
    print("\n\n=========\nImages shape: ", output["images"][camera_indices[0]].shape)

    # Normalize states and actions to [-1, 1]
    obs_min = np.min(output["states"], axis=0)
    obs_max = np.max(output["states"], axis=0)
    action_min = np.min(output["actions"], axis=0)
    action_max = np.max(output["actions"], axis=0)
    output["raw_states"] = output["states"].copy()
    output["raw_actions"] = output["actions"].copy()
    output["states"] = 2 * (output["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
    output["actions"] = (
        2 * (output["actions"] - action_min) / (action_max - action_min + 1e-6) - 1
    )
    print("States min (after normalization):", np.min(output["states"], axis=0))
    print("States max (after normalization):", np.max(output["states"], axis=0))
    print("Actions min (after normalization):", np.min(output["actions"], axis=0))
    print("Actions max (after normalization):", np.max(output["actions"], axis=0))

    # Get min and max of state-action difference
    state_action_diff_min = np.min(np.stack(state_action_diff_mins), axis=0)
    state_action_diff_max = np.max(np.stack(state_action_diff_maxs), axis=0)

    # Configure dataset name based on keys
    dataset_name = ""
    if "cartesian_position" in observation_keys:
        dataset_name += "eef"
    if "joint_positions" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "js"
    if "joint_velocities" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "jv"
    if "gripper_position" in observation_keys:
        dataset_name += "g"
    dataset_name += "_"
    if "cartesian_position" in action_keys:
        dataset_name += "eef"
    if "joint_position" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "js"
    if "joint_velocities" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "jv"
    if "gripper_position" in action_keys:
        dataset_name += "g"
    if save_image:
        dataset_name += "_"
        dataset_name += f"{len(camera_indices)}cam"
        dataset_name += f"_{img_resolution[0]}"
    if additional_name:
        dataset_name += f"_{additional_name}"

    # Create output directory
    output_dir = os.path.join(output_parent_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config into a text file
    config = {
        "action_keys": action_keys,
        "observation_keys": observation_keys,
        "img_resolution": img_resolution,
        "camera_indices": camera_indices,
        "bgr2rgb": bgr2rgb,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "action_min": action_min,
        "action_max": action_max,
        "delta_min": state_action_diff_min,
        "delta_max": state_action_diff_max,
        "num_traj": len(traj_paths),
    }
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Save the normalization values and processed dataset
    np.savez(
        os.path.join(output_dir, "norm.npz"),
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        delta_min=state_action_diff_min,
        delta_max=state_action_diff_max,
    )
    np.savez_compressed(
        os.path.join(output_dir, "dataset.npz"),
        **output,
    )
    print("Data and normalization values saved in", output_dir)


def process_real_dataset_with_sim_data(
    input_paths,
    output_parent_dir,
    num_traj=None,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    horizon_steps=16,
    img_resolution=(96, 96),
    camera_indices=["0", "2"],
    num_thread=10,
    skip_image=False,
    keep_bgr=False,
    use_obs_as_action=False,
    obs_gripper_threshold=0.2,
    additional_name="",
    sim_data_ratio=1.0,
):
    save_image = not skip_image
    bgr2rgb = not keep_bgr

    # concatenate all paths
    real_traj_paths = []
    sim_traj_paths = []
    for path in input_paths:
        if "sim" in path:
            sim_traj_paths += [
                os.path.join(path, traj_name)
                for traj_name in os.listdir(path)
                if traj_name.endswith(".h5") and "failed" not in traj_name
            ]
        else:
            real_traj_paths += [
                os.path.join(path, traj_name)
                for traj_name in os.listdir(path)
                if traj_name.endswith(".h5")
            ]
    # Random sample sim data
    num_sim_traj = int(len(sim_traj_paths) * sim_data_ratio)
    sim_traj_paths = np.random.choice(sim_traj_paths, num_sim_traj, replace=False)

    traj_paths = real_traj_paths + list(sim_traj_paths)
    num_traj_available = len(traj_paths)

    # process first trajectories
    if num_traj is not None:
        traj_paths = traj_paths[:num_traj]
    else:
        num_traj = num_traj_available
    print(
        f"Processing {num_traj}/{num_traj_available} trajectories with {cpu_count()} cpu threads..."
    )

    # initialize output dictionary
    output = {
        "traj_lengths": [],
        "actions": [],
        "states": [],
        "images": {index: [] for index in camera_indices},
    }
    state_action_diff_mins = []
    state_action_diff_maxs = []
    for traj_path in tqdm(traj_paths):
        # load trajectory from h5
        s1 = time.time()
        if "sim" in traj_path:
            print("Loading sim data")
            traj, camera_indices_raw = load_sim_hdf5_for_training(
                traj_path,
                action_keys=action_keys,
                observation_keys=observation_keys,
                load_image=save_image,
            )
            traj["action/gripper_position"] = 1 - (traj["action/gripper_position"] + 1) / 2
            traj["observation/robot_state/gripper_position"] = 1 - (
                traj["observation/robot_state/gripper_position"] / 0.04
            )
        else:
            traj, camera_indices_raw = load_hdf5(
                traj_path,
                action_keys=action_keys,
                observation_keys=observation_keys,
                load_image=save_image,
            )
        print("Time to load h5:", time.time() - s1)

        # skip idle actions (skip_action == True)
        keep_idx = ~traj["observation/timestamp/skip_action"]
        traj_length = len(keep_idx)
        output["traj_lengths"].append(traj_length)
        print("Path:", traj_path, "Length:", traj_length)

        # set gripper position to binary: 0 for open, 1 for closed
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = (
                traj["action/gripper_position"] > 0.5
            ).astype(np.float32)
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = (
                traj["observation/robot_state/gripper_position"] > obs_gripper_threshold
            ).astype(np.float32)

        # set roll to always be positive
        if "cartesian_position" in action_keys:
            traj["action/cartesian_position"][:, 3] = np.abs(
                traj["action/cartesian_position"][:, 3]
            )
        if "cartesian_position" in observation_keys:
            traj["observation/robot_state/cartesian_position"][:, 3] = np.abs(
                traj["observation/robot_state/cartesian_position"][:, 3]
            )

        # add dimension to gripper position
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = traj["action/gripper_position"][:, None]
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = traj[
                "observation/robot_state/gripper_position"
            ][:, None]

        # get the maximum difference between the starting state and each action of the chunk at each timestep
        if "joint_positions" in observation_keys and "joint_position" in action_keys:
            state = traj["observation/robot_state/joint_positions"][keep_idx]
            action = traj["action/joint_position"][keep_idx]
        elif (
            "cartesian_position" in observation_keys
            and "cartesian_position" in action_keys
        ):
            state = traj["observation/robot_state/cartesian_position"][keep_idx]
            action = traj["action/cartesian_position"][keep_idx]
        else:
            raise NotImplementedError(
                "For getting the state-action difference, need consistent keys"
            )
        diffs = np.empty((0, action.shape[1]))
        for step in range(horizon_steps // 4, horizon_steps):  # skip early steps
            diff = action[step:] - state[:-step]
            diffs = np.concatenate([diffs, diff], axis=0)
        state_action_diff_mins.append(np.min(diffs, axis=0))
        state_action_diff_maxs.append(np.max(diffs, axis=0))
        if np.isnan(np.sum(diffs)) > 0 or np.isnan(np.sum(diffs)) > 0:
            raise ValueError("NaN in state-action difference")

        # concatenate states and actions
        if use_obs_as_action:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][
                        :-1
                    ]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx][1:]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            output["traj_lengths"][-1] = output["traj_lengths"][-1] - 1
        else:
            states = np.concatenate(
                [
                    traj[f"observation/robot_state/{observation_keys[i]}"][keep_idx]
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            actions = np.concatenate(
                [
                    traj[f"action/{action_keys[i]}"][keep_idx]
                    for i in range(len(action_keys))
                ],
                axis=1,
            )
        assert len(states) == output["traj_lengths"][-1]
        assert len(actions) == output["traj_lengths"][-1]
        output["states"].append(states)
        output["actions"].append(actions)

        # add images
        if save_image:
            # verify camera indices
            camera_indices_chosen = [camera_indices_raw[idx] for idx in camera_indices]
            print(f"Using raw camera indices: {camera_indices_chosen}")
            for raw_idx, idx in zip(camera_indices_chosen, camera_indices):
                raw_img = traj[f"observation/image/{raw_idx}"][keep_idx]  # (T, H, W, C)
                assert raw_img.dtype == np.uint8

                # resize with multiprocessing
                s1 = time.time()
                resized_img = resize_images_multiprocessing(
                    raw_img,
                    img_resolution,
                    num_thread,
                )
                print("Time to resize images:", time.time() - s1)

                # Transpose to (T, C, H, W)
                resized_img = resized_img.transpose(0, 3, 1, 2)

                # Change BGR (cv2 default) to RGB
                if bgr2rgb:
                    if "sim" not in traj_path:
                        resized_img = resized_img[:, [2, 1, 0]]

                if use_obs_as_action:
                    resized_img = resized_img[:-1]

                # save
                assert len(resized_img) == output["traj_lengths"][-1]
                output["images"][idx].append(resized_img)

    # Convert to numpy arrays
    output["traj_lengths"] = np.array(output["traj_lengths"])
    output["actions"] = np.concatenate(output["actions"], axis=0)
    output["states"] = np.concatenate(output["states"], axis=0)

    for idx in camera_indices:
        output["images"][idx] = np.concatenate(output["images"][idx], axis=0)
    print("\n\n=========\nImages shape: ", output["images"][camera_indices[0]].shape)

    # Normalize states and actions to [-1, 1]
    obs_min = np.min(output["states"], axis=0)
    obs_max = np.max(output["states"], axis=0)
    action_min = np.min(output["actions"], axis=0)
    action_max = np.max(output["actions"], axis=0)
    output["raw_states"] = output["states"].copy()
    output["raw_actions"] = output["actions"].copy()
    output["states"] = 2 * (output["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
    output["actions"] = (
        2 * (output["actions"] - action_min) / (action_max - action_min + 1e-6) - 1
    )
    print("States min (after normalization):", np.min(output["states"], axis=0))
    print("States max (after normalization):", np.max(output["states"], axis=0))
    print("Actions min (after normalization):", np.min(output["actions"], axis=0))
    print("Actions max (after normalization):", np.max(output["actions"], axis=0))

    # Get min and max of state-action difference
    state_action_diff_min = np.min(np.stack(state_action_diff_mins), axis=0)
    state_action_diff_max = np.max(np.stack(state_action_diff_maxs), axis=0)

    # Configure dataset name based on keys
    dataset_name = ""
    if "cartesian_position" in observation_keys:
        dataset_name += "eef"
    if "joint_positions" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "js"
    if "joint_velocities" in observation_keys:
        assert "cartesian_position" not in observation_keys
        dataset_name += "jv"
    if "gripper_position" in observation_keys:
        dataset_name += "g"
    dataset_name += "_"
    if "cartesian_position" in action_keys:
        dataset_name += "eef"
    if "joint_position" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "js"
    if "joint_velocities" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "jv"
    if "gripper_position" in action_keys:
        dataset_name += "g"
    if save_image:
        dataset_name += "_"
        dataset_name += f"{len(camera_indices)}cam"
        dataset_name += f"_{img_resolution[0]}"
    dataset_name += f"_{additional_name}_sim_{sim_data_ratio}"

    # Create output directory
    output_dir = os.path.join(output_parent_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config into a text file
    config = {
        "action_keys": action_keys,
        "observation_keys": observation_keys,
        "img_resolution": img_resolution,
        "camera_indices": camera_indices,
        "bgr2rgb": bgr2rgb,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "action_min": action_min,
        "action_max": action_max,
        "delta_min": state_action_diff_min,
        "delta_max": state_action_diff_max,
        "num_traj": len(traj_paths),
    }
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Save the normalization values and processed dataset
    np.savez(
        os.path.join(output_dir, "norm.npz"),
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        delta_min=state_action_diff_min,
        delta_max=state_action_diff_max,
    )
    np.savez_compressed(
        os.path.join(output_dir, "dataset.npz"),
        **output,
    )
    print("Data and normalization values saved in", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in",
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_parent_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num_traj",
        type=int,
        default=None,  # None for processing all available ones
    )
    parser.add_argument(
        "-t",
        "--num_thread",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-c",
        "--camera_indices",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Raw data uses index from cv2, which might be something like [2, 4, 8]. We will use the index from the raw data, in this case, 0, 1, 2",
    )
    parser.add_argument(
        "-res",
        "--img_resolution",
        type=int,
        nargs=2,
        default=[192, 192],
    )
    parser.add_argument(
        "-a",
        "--action_keys",
        type=str,
        nargs="+",
        default=[
            "joint_position",
            "gripper_position",
        ],  # "cartesian_position"
    )
    parser.add_argument(
        "-o",
        "--observation_keys",
        type=str,
        nargs="+",
        default=[
            "joint_positions",
            "gripper_position",
        ],  # "joint_velocities", "cartesian_positions"
    )
    parser.add_argument(
        "-tp",
        "--horizon_steps",
        type=int,
        default=16,  # we are not saving action chunks, but to get the maximum difference between the state and each step of action, for delta actions
    )
    parser.add_argument(
        "--skip_image",
        action="store_true",
    )
    parser.add_argument(
        "--keep_bgr",
        action="store_true",
    )
    parser.add_argument(
        "--use_obs_as_action",
        action="store_true",
    )
    parser.add_argument(
        "--obs_gripper_threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--additional_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--sim_data_ratio",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    with_sim_data = False
    for path in args.input_paths:
        if "sim" in path:
            with_sim_data = True
            break
    if with_sim_data:
        process_real_dataset_with_sim_data(**vars(args))
    else:
        process_real_dataset(**vars(args))
