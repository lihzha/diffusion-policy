import h5py
import numpy as np
import os
import cv2


def load_hdf5_to_dict(
    hdf5_file_path,
    action_keys=["cartesian_position", "gripper_position"],  # "joint_positions"
    observation_keys=["joint_positions", "gripper_position"],  # "joint_velocities"
    use_image=True,
):
    """
    Load an HDF5 file into a nested dictionary, including only specified keys.

    Parameters:
    hdf5_file_path (str): Path to the HDF5 file.

    Returns:
    dict: A dictionary representation of the HDF5 file.
    """
    # Define keys to load in hierarchical form

    keys_to_load = ["observation/timestamp/skip_action"]

    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if use_image:
        keys_to_load.append("observation/image")

    def should_load(path):
        """Check if the current path should be loaded based on the specified keys."""
        return any(path == key or path.startswith(f"{key}/") for key in keys_to_load)

    def recursive_load(group, path=""):
        """Recursively load a group into a dictionary, including only specified keys."""
        result = {}
        for key, item in group.items():
            current_path = f"{path}/{key}".strip("/")
            if isinstance(item, h5py.Group):
                # If the item is a group, recurse into it
                nested_result = recursive_load(item, current_path)
                if nested_result:  # Only add if there's valid data in the nested result
                    result[key] = nested_result
            else:
                # If the item is a dataset, store it as a numpy array if it's a key to load
                if should_load(current_path):
                    result[key] = item[()]
        return result

    with h5py.File(hdf5_file_path, "r") as file:
        return recursive_load(file)


def process_real_dataset(
    dataset_path,
    use_image=True,
    action_keys=["joint_position", "gripper_position"],  # "cartesian_position"
    observation_keys=[
        "joint_positions",
        "gripper_position",
    ],  # "joint_velocities", "cartesian_position"):
    img_resolution=(96, 96),  # (H, W)
    bgr2rgb=True,
    camera_idx = ["0", "3"]  #'2' is right camera, '0' is left camera, '3' is wrist camera
):

    dataset = {"traj_lengths": [], "actions": [], "states": [], "images": {key: [] for key in camera_idx}}

    for traj_idx, file in enumerate(os.listdir(dataset_path)):
        if file.endswith(".h5"):
            traj = load_hdf5_to_dict(
                os.path.join(dataset_path, file),
                action_keys=action_keys,
                observation_keys=observation_keys,
                use_image=use_image,
            )

            keep_idx = ~traj["observation"]["timestamp"]["skip_action"]
            dataset["traj_lengths"].append(np.sum(keep_idx))
            
            if "gripper_position" in action_keys:
                for i in range(len(traj["action"]["gripper_position"])):
                    if traj["action"]["gripper_position"][i] <= 0.5:
                        traj["action"]["gripper_position"][i] = 0.0
                    else:
                        traj["action"]["gripper_position"][i] = 1.0

            if "cartesian_position" in action_keys:
                for i in range(len(traj["action"]["cartesian_position"])):
                    traj["action"]["cartesian_position"][i, 3] = np.abs(
                        traj["action"]["cartesian_position"][i, 3]
                    )  # flip roll angle

            dataset["actions"].append(
                np.concatenate(
                    [
                        (
                            traj["action"][action_keys[i]][keep_idx]
                            if action_keys[i] != "gripper_position"
                            else traj["action"][action_keys[i]][keep_idx, None]
                        )
                        for i in range(len(action_keys))
                    ],
                    axis=1,
                )
            )

            if "gripper_position" in observation_keys:
                for i in range(
                    len(traj["observation"]["robot_state"]["gripper_position"])
                ):
                    if (
                        traj["observation"]["robot_state"]["gripper_position"][i]
                        <= 0.5
                    ):
                        traj["observation"]["robot_state"]["gripper_position"][i] = 0.0
                    else:
                        traj["observation"]["robot_state"]["gripper_position"][i] = 1.0

            if "cartesian_position" in observation_keys:
                for i in range(
                    len(traj["observation"]["robot_state"]["cartesian_position"])
                ):
                    traj["observation"]["robot_state"]["cartesian_position"][i, 3] = (
                        np.abs(
                            traj["observation"]["robot_state"]["cartesian_position"][
                                i, 3
                            ]
                        )
                    )

            states = np.concatenate(
                [
                    (
                        traj["observation"]["robot_state"][observation_keys[i]][keep_idx]
                        if observation_keys[i] != "gripper_position"
                        else traj["observation"]["robot_state"][observation_keys[i]][
                            keep_idx, None
                        ]
                    )
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            
            assert len(states) == dataset["traj_lengths"][-1]
            assert len(dataset["actions"][-1]) == dataset["traj_lengths"][-1]
            
            dataset["states"].append(states)
            
            if use_image:
                assert isinstance(traj["observation"]["image"], dict), "Only support multiple camera images"
                for idx in camera_idx:                    
                    raw_img = traj["observation"]["image"][idx][keep_idx]  # (T, H, W, C)
                    resized_img = np.empty(
                        (raw_img.shape[0], 3, *img_resolution), dtype=raw_img.dtype
                    )
                    for i in range(raw_img.shape[0]):
                        resized_img[i] = cv2.resize(
                            raw_img[i], (img_resolution[1], img_resolution[0])  # (W, H)
                        ).transpose(2, 0, 1)
                        # From BGR to RGB
                        if bgr2rgb:
                            resized_img[i] = resized_img[i][[2, 1, 0], :, :]
                    dataset["images"][idx].append(resized_img.copy())
                    assert len(dataset["images"][idx][-1]) == dataset["traj_lengths"][-1]
                

    print(f"Loaded {len(dataset['traj_lengths'])} trajectories")
    dataset["traj_lengths"] = np.array(dataset["traj_lengths"])
    dataset["actions"] = np.concatenate(dataset["actions"], axis=0)
    dataset["states"] = np.concatenate(dataset["states"], axis=0)
    for idx in camera_idx:
        dataset["images"][idx] = np.concatenate(dataset["images"][idx], axis=0)
    print("Images shape: ", dataset["images"][camera_idx[0]].shape)
        
    # Normalize states and actions to [-1, 1]
    obs_min = np.min(dataset["states"], axis=0)
    obs_max = np.max(dataset["states"], axis=0)
    action_min = np.min(dataset["actions"], axis=0)
    action_max = np.max(dataset["actions"], axis=0)
    dataset["states"] = (
        2 * (dataset["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
    )
    dataset["actions"] = (
        2 * (dataset["actions"] - action_min) / (action_max - action_min + 1e-6) - 1
    )

    # Save normalization values to .npz
    
    
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
    if "joint_positions" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "js"
    if "joint_velocities" in action_keys:
        assert "cartesian_position" not in action_keys
        dataset_name += "jv"
    if "gripper_position" in action_keys:
        dataset_name += "g"
    
    if use_image:
        dataset_name += "_"
        dataset_name += f"{len(camera_idx)}cam"
        dataset_name += f"_{img_resolution[0]}"
    
        
    np.savez(
        os.path.join(os.path.dirname(dataset_path), f"norm_{dataset_name}.npz"),
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )
    # Save the dataset as a compressed numpy file
    np.savez_compressed(
        os.path.join(os.path.dirname(dataset_path), f"dataset_{dataset_name}.npz"),
        **dataset,
    )


def get_no_vel_dataset(dataset_path):
    processed_dataset_path = os.path.join(dataset_path, "processed_dataset.npz")
    processed_norm_path = os.path.join(dataset_path, "norm.npz")

    dataset = np.load(processed_dataset_path, allow_pickle=True)
    states = dataset["states"]
    states = np.concatenate([states[:, :7], states[:, -1:]], axis=1)
    new_dataset = {
        key: dataset[key] if key != "states" else states for key in dataset.files
    }

    np.savez_compressed(
        os.path.join(
            os.path.dirname(processed_dataset_path), "processed_dataset_no_vel.npz"
        ),
        **new_dataset,
    )

    norm = np.load(processed_norm_path)
    new_norm = {
        key: (
            norm[key]
            if "obs" not in key
            else np.concatenate([norm[key][:7], norm[key][-1:]])
        )
        for key in norm.files
    }
    np.savez(
        os.path.join(os.path.dirname(processed_norm_path), "norm_no_vel.npz"),
        **new_norm,
    )


if __name__ == "__main__":
    dataset_path = "/home/lab/guided-data-collection/data/tomato_plate_trials-date?"
    process_real_dataset(dataset_path, use_image=True, camera_idx=["0"], bgr2rgb=True, img_resolution=(196, 196), 
                         action_keys=["joint_position", "gripper_position"], observation_keys=["joint_positions", "gripper_position"])
    
    
    # get_no_vel_dataset(dataset_path)
    # d = np.load("/home/lab/droid/traj_data/norm_no_vel.npz", allow_pickle=True)
    # print(d['obs_min'].shape)
