import torch
import numpy as np

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions, supports both list/numpy array and torch.Tensor input.

    Parameters:
    q1 : list, numpy array, or torch.Tensor
        First quaternion [w1, x1, y1, z1]
    q2 : list, numpy array, or torch.Tensor
        Second quaternion [w2, x2, y2, z2]

    Returns:
    result : list or torch.Tensor
        Resultant quaternion after multiplication [w, x, y, z]
        Output is on GPU if input is torch.Tensor on GPU.
    """

    if isinstance(q1, torch.Tensor):
        # Ensure the inputs are torch tensors
        if q1.device != q2.device:
            raise ValueError("q1 and q2 must be on the same device (CPU or GPU).")

        # Extract components of the quaternions
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Perform quaternion multiplication
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.stack([w, x, y, z], dim=-1)
    
    elif isinstance(q1, (list, np.ndarray, tuple)):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return np.array([w, x, y, z])
    
    else:
        raise NotImplementedError