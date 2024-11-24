import random
import torch
import numpy as np

def randomize_continuous(min, max, size: int, return_list=False):
    '''
    Return a list with length 'size', with each element's value within the interval [min, max].
    '''
    if size > 1:
        return [random.uniform(min, max) for _ in range(size)]
    else:
        if return_list:
            return [random.uniform(min, max)]
        else:
            return random.uniform(min, max)

def randomize_by_percentage(array, low_percentage=0.9, high_percentage=1.1):
    """
    Randomizes elements of the input array/tensor by multiplying them with
    a random factor between (low_percentage) and (high_percentage).
    
    Parameters:
    - array: A numpy array, torch tensor, or list containing the values to be randomized.
    - low_percentage: The lower bound of the percentage change (e.g., -0.1 for -10%).
    - high_percentage: The upper bound of the percentage change (e.g., 1.1 for +110%).
    
    Returns:
    - randomized_array: An array/tensor/list of the same shape with randomized values.
    """
    
    # Check if the input is a torch tensor
    if isinstance(array, torch.Tensor):
        device = array.device  # Store the device (e.g., GPU or CPU)
        # Generate random factors on the same device as the input tensor
        random_factors = torch.empty_like(array).uniform_(low_percentage, high_percentage).to(device)
        randomized_array = array * random_factors
        
    # Check if the input is a numpy array
    elif isinstance(array, np.ndarray):
        # Generate random factors as a numpy array
        random_factors = np.random.uniform(low_percentage, high_percentage, size=array.shape)
        randomized_array = array * random_factors
    
    # If it's a list, process as numpy array without conversion
    elif isinstance(array, list):
        array = np.array(array)
        random_factors = np.random.uniform(low_percentage, high_percentage, size=array.shape)
        randomized_array = array * random_factors
    
    elif isinstance(array, float):
        random_factors = np.random.uniform(low_percentage, high_percentage, size=1).item()
        randomized_array = array * random_factors
    
    return randomized_array

def randomize_uv(top_uv, aspect_ratio, shift_range=0.1):
    """
    Randomizes the UV coordinates for a periodic texture, scaling for non-square images.

    Parameters:
    - top_uv (np.array): The original UV coordinates to be randomized.
    - image_width (int): The width of the texture image.
    - image_height (int): The height of the texture image.
    - shift_range (float): The maximum shift value for randomization.

    Returns:
    - np.array: The randomized UV coordinates.
    """

    # Determine the range of UV coordinates
    uv_min, uv_max = top_uv.min(axis=0), top_uv.max(axis=0)
    step_size = uv_max - uv_min

    # Adjust step size for aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        step_size[1] *= aspect_ratio
    else:  # Taller than wide
        step_size[0] /= aspect_ratio

    # Randomize the starting point
    random_shift = np.random.uniform(0, shift_range, size=2)

    # Shift the UV coordinates
    randomized_uv = top_uv - uv_min
    randomized_uv = randomized_uv / step_size
    randomized_uv = (randomized_uv + random_shift) % 1.0  # Wrap around the texture space

    return randomized_uv