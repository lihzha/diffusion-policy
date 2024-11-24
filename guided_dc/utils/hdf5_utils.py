import numpy as np
import torch
import h5py

def to_cpu(var):
    if isinstance(var, torch.Tensor):
        return var.cpu()  # Returns True if tensor is on GPU, False otherwise
    return var  # If not a torch tensor, return False

def save_dict_to_hdf5(hdf5_group, data_dict):
    """Recursively saves a nested dictionary to an HDF5 group."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a subgroup for the nested dictionary
            subgroup = hdf5_group.create_group(key)
            save_dict_to_hdf5(subgroup, value)  # Recursively save the nested dict
        else:
            # Save the value as a dataset
            if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                value = np.array(to_cpu(value))  # Convert lists/tuples to numpy arrays
            hdf5_group.create_dataset(key, data=value)

def save_extra_info_to_hdf5(env, kwargs):
    traj_id = "traj_{}".format(env._episode_id)
    group = env._h5_file[traj_id]
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            value = np.array(to_cpu(value))
            group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            group.create_group(key)
            save_dict_to_hdf5(group[key], to_cpu(value))
        elif isinstance(value, str):
            # Create a dataset with variable-length strings
            dt = h5py.string_dtype(encoding='utf-8')
            group.create_dataset(key, data=value, dtype=dt)