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
    
    keys_to_load = []
   
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


def process_real_dataset(dataset_path="/home/lab/droid/traj_data/", 
                         use_image=True,
                         action_keys = ["cartesian_position", "gripper_position"],  # "cartesian_position"
                         observation_keys = ["cartesian_position", "gripper_position"], # "joint_velocities", "cartesian_position"):
            ):  
    
    dataset = {"traj_lengths": [], "actions": [], "states": [], "images": []}
    
    for i, file in enumerate(os.listdir(dataset_path)):
        if file.endswith(".h5"):
            _data = load_hdf5_to_dict(dataset_path + file, action_keys=action_keys, 
                                      observation_keys=observation_keys, use_image=use_image)
            
            dataset["traj_lengths"].append(len(_data["action"][action_keys[0]]))
            
            if "gripper_position" in action_keys:
                for i in range(len(_data["action"]["gripper_position"])):
                    if _data["action"]["gripper_position"][i] <= 0.5 :
                        _data["action"]["gripper_position"][i] = 0.
                    else:
                        _data["action"]["gripper_position"][i] = 1.
            
            if "cartesian_position" in action_keys:
                for i in range(len(_data["action"]["cartesian_position"])):
                    _data["action"]["cartesian_position"][i,3] = np.abs(_data["action"]["cartesian_position"][i,3])  # flip roll angle
            
            dataset["actions"].append(
                np.concatenate(
                    [
                        _data["action"][action_keys[i]] if action_keys[i] != "gripper_position" else _data["action"][action_keys[i]][:, None] 
                        for i in range(len(action_keys))
                    ],
                    axis=1,
                )
            )
            
            if "gripper_position" in observation_keys:
                for i in range(len( _data["observation"]["robot_state"]["gripper_position"])):
                    if  _data["observation"]["robot_state"]["gripper_position"][i] <= 0.5 :
                        _data["observation"]["robot_state"]["gripper_position"][i] = 0.
                    else:
                        _data["observation"]["robot_state"]["gripper_position"][i] = 1.
            
            if "cartesian_position" in observation_keys:
                for i in range(len(_data["observation"]["robot_state"]["cartesian_position"])):
                    _data["observation"]["robot_state"]["cartesian_position"][i,3] = np.abs(_data["observation"]["robot_state"]["cartesian_position"][i,3])
                
            states = np.concatenate(
                [
                    _data["observation"]["robot_state"][observation_keys[i]] if observation_keys[i] != "gripper_position" else _data["observation"]["robot_state"][observation_keys[i]][:, None] 
                    for i in range(len(observation_keys))
                ],
                axis=1,
            )
            dataset["states"].append(states)
            if use_image:
                dataset["images"].append(_data["observation"]["image"])
            

    print(f"Loaded {len(dataset['traj_lengths'])} trajectories")
    # print memory allocations to store the dataset
    # print(f"Memory usage: {sum([sys.getsizeof(v) for v in dataset.values()]) / 1e6} MB")
    dataset["traj_lengths"] = np.array(dataset["traj_lengths"])
    dataset["actions"] = np.concatenate(dataset["actions"], axis=0)
    dataset["states"] = np.concatenate(dataset["states"], axis=0)
    
    if use_image:
    
        if isinstance(_data["observation"]["image"], dict):

            camera_idx = ['0', '3']   #'2' is right camera, '0' is left camera, '3' is wrist camera
            images = {}
            for idx in camera_idx:
                # Reshape images to (num_steps, C, H, W)
                images[idx] = np.concatenate(
                    [_data[idx] for _data in dataset["images"]], axis=0
                )

                # if images[idx].shape[-1] == 3:
                #     images[idx] = np.transpose(images[idx], (0, 3, 1, 2))
                # else:
                #     assert images[idx].shape[-1] == 3, "Only support RGB images"

                resized_images = np.empty(
                    (images[idx].shape[0], 3, 240, 320), dtype=images[idx].dtype
                )
                for i in range(images[idx].shape[0]):
                    resized_images[i] = cv2.resize(images[idx][i], (320, 240)).transpose(
                        2, 0, 1
                    )
                images[idx] = resized_images

            dataset["images"] = images
        else:
            raise ValueError("Only support multiple camera images")

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
    np.savez(
        os.path.join(os.path.dirname(dataset_path), "norm_eefg_eefg_2cam.npz"),
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )
    # Save the dataset as a compressed numpy file
    np.savez_compressed(
        os.path.join(os.path.dirname(dataset_path), "dataset_eefg_eefg_2cam.npz"), **dataset
    )

def get_no_vel_dataset(dataset_path):
    processed_dataset_path = os.path.join(dataset_path, "processed_dataset.npz")
    processed_norm_path = os.path.join(dataset_path, "norm.npz")

    dataset = np.load(processed_dataset_path, allow_pickle=True)
    states = dataset["states"]
    states = np.concatenate([states[:, :7], states[:, -1:]], axis=1)
    new_dataset = {key: dataset[key] if key != 'states' else states for key in dataset.files}

    np.savez_compressed(
        os.path.join(os.path.dirname(processed_dataset_path), "processed_dataset_no_vel.npz"),
        **new_dataset
    )

    norm = np.load(processed_norm_path)
    new_norm = {key: norm[key] if 'obs' not in key else np.concatenate([norm[key][:7], norm[key][-1:]]) for key in norm.files}
    np.savez(
        os.path.join(os.path.dirname(processed_norm_path), "norm_no_vel.npz"),
        **new_norm
    )


if __name__ == '__main__':
    dataset_path = "/home/lab/droid/traj_data/"
    process_real_dataset(dataset_path)
    # get_no_vel_dataset(dataset_path)
    # d = np.load("/home/lab/droid/traj_data/norm_no_vel.npz", allow_pickle=True)
    # print(d['obs_min'].shape)