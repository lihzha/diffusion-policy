import numpy as np
from scipy.spatial.transform import Rotation as R


def get_sim_camera_global_pose_from_real(
    robot_base_global_pose, camera_robot_base_pose
):
    # Unpack inputs
    pos_b, quat_b = (
        np.array(robot_base_global_pose[:3]),
        np.array(robot_base_global_pose[3:]),
    )
    pos_c, quat_c = (
        np.array(camera_robot_base_pose[:3]),
        np.array(camera_robot_base_pose[3:]),
    )

    # Normalize quaternions
    quat_b /= np.linalg.norm(quat_b)
    quat_c /= np.linalg.norm(quat_c)

    quat_b = np.roll(quat_b, -1)  # from w, x, y, z to x, y, z, w
    quat_c = np.roll(quat_c, -1)

    # Convert quaternion to rotation matrix for base
    R_b = R.from_quat(quat_b).as_matrix()  # Quaternion format: [x, y, z, w]

    # Debugging: Print rotation matrix
    print("Rotation matrix R_b:\n", R_b)

    # Transform position
    pos_g = pos_b + R_b @ pos_c

    # Debugging: Print intermediate positions
    print("Base global position (pos_b):", pos_b)
    print("Camera relative position (pos_c):", pos_c)
    print("Transformed global position (pos_g):", pos_g)

    # Transform orientation (quaternion multiplication)
    quat_b_rot = R.from_quat(quat_b)
    quat_c_rot = R.from_quat(quat_c)
    quat_g_rot_oepncv = quat_b_rot * quat_c_rot  # Quaternion multiplication

    # from x, y, z, w to w, x, y, z
    # quat_g_opencv = np.roll(quat_g, 1)

    opencv2ros_rot = R.from_euler("xyz", [90, -90, 0], degrees=True)

    quat_g_rot_ros = quat_g_rot_oepncv * opencv2ros_rot

    quat_g = quat_g_rot_ros.as_quat()
    quat_g = np.roll(quat_g, 1)

    print("Resulting global quaternion (quat_g):", quat_g)
    return np.concatenate([pos_g, quat_g])


if __name__ == "__main__":
    # Example input
    robot_base_global_pose = [
        -0.547,
        -0.527,
        -0.143,
        0.9238795325127119,
        -0.0,
        -0.0,
        0.3826834323616491,
    ]
    camera_robot_base_pose = [
        0.40794,
        0.71129,
        0.73127,
        0.051215,
        -0.15300,
        0.90863,
        -0.385174,
    ]

    # Compute camera global pose
    camera_global_pose = get_sim_camera_global_pose_from_real(
        robot_base_global_pose, camera_robot_base_pose
    )
    print("Camera's pose in the global frame:", list(camera_global_pose))
