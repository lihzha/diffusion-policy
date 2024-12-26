"""
In order to calibrate the wrist camera in simulation, so that it's aligned with the real wrist camera, we need to follow the steps below:
1. Get a real rgbd image: since realsense's color and depth camera are not aligned and have different intrinsics,
    we need to (1) make sure their resolution match, (2) align the depth image with the color image using Realsense SDK.

2. Get a simulation rgbd image: set the resolution and intrinsics to be the same as the real color camera. Take an image that is
    roughly the same as the real image.

3. Postprocess the real rgbd image by multiplying depth_scale (0.0001) and the convert to milimeters.
4. Segment only the hand and gripper part in both real and simulation rgb images.
5. Use the segmented images to first convert rgbd to point cloud, then align the point cloud from simulation to the real one.
6. Get the transformation matrix from the above step and apply it to the wrist camera in simulation.
"""

import logging
import os

import cv2
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mani_skill.utils.structs import Pose
from omegaconf import OmegaConf
from PIL import Image

import guided_dc  # noqa: F401

OmegaConf.register_resolver(
    "pi_op",
    lambda operation, expr=None: {
        "div": np.pi / eval(expr) if expr else np.pi,
        "mul": np.pi * eval(expr) if expr else np.pi,
        "raw": np.pi,
    }[operation],
)

log = logging.getLogger(__name__)

# eye = [-0.3662, -0.1873, 0.3187]
# target = [-0.20, -0.37, 0]
init_pose = Pose.create_from_pq(
    p=[-0.3682, -0.1811, 0.3338], q=[0.8211, 0.1625, 0.4241, -0.3456]
)


def visualize_segmentation(segmentation_image):
    """
    Visualize a segmentation image with a color map.

    Args:
        segmentation_image (numpy array): Segmentation image of shape (H, W, 1) or (H, W),
                                           where each pixel value is the object class ID.
    """
    # Define a color map for visualization
    num_classes = np.max(segmentation_image) + 1  # Assuming class IDs are 0-based
    colors = plt.cm.get_cmap("tab20", num_classes)  # Use a predefined colormap

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(segmentation_image.squeeze(), cmap=colors, interpolation="nearest")
    plt.colorbar(ticks=range(num_classes), label="Class ID")
    plt.title("Segmentation Visualization")
    plt.axis("off")
    plt.show()


def process_segmentation(segmentation_image, target_classes):
    """
    Process the segmentation image by setting specified class IDs to 1 and others to 0.

    Args:
        segmentation_image (numpy array): Segmentation image of shape (H, W, 1) or (H, W).
        target_classes (list): List of class IDs to set to 1.

    Returns:
        numpy array: Processed binary segmentation image of shape (H, W).
    """
    # Create a binary mask
    binary_segmentation = np.isin(segmentation_image.squeeze(), target_classes).astype(
        np.uint8
    )
    return binary_segmentation


def visualize_binary_segmentation(binary_segmentation):
    """
    Visualize a binary segmentation image.

    Args:
        binary_segmentation (numpy array): Binary segmentation image of shape (H, W).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_segmentation, cmap="gray", interpolation="nearest")
    plt.title("Processed Binary Segmentation")
    plt.axis("off")
    plt.show()


def get_sim_rgbd(env, visualize=False):
    # pose = sapien_utils.look_at(eye=eye, target=target)
    # env.wrist_mount.set_pose(pose)

    env.wrist_mount.set_pose(init_pose)

    # Reset the environment and visualize the observation
    obs, _ = env.reset()
    rgb_image = (
        obs["sensor_data"]["hand_camera"]["rgb"]
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.uint8)
    )
    depth_image = (
        obs["sensor_data"]["hand_camera"]["depth"]
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.uint8)
    )

    # rgb_image = cv2.resize(rgb_image, (320, 240))
    print(depth_image.shape, rgb_image.shape)
    cv2.imwrite("calibration/sim/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    np.save("calibration/sim/depth.npy", depth_image)
    color_mapped_depth = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    cv2.imwrite("calibration/sim/depth.png", color_mapped_depth)
    if visualize:
        visualize_segmentation(
            obs["sensor_data"]["hand_camera"]["segmentation"].cpu().numpy()[0]
        )
    segmentation = process_segmentation(
        obs["sensor_data"]["hand_camera"]["segmentation"].cpu().numpy()[0],
        target_classes=[10, 12, 13],
    ).squeeze()
    if visualize:
        visualize_binary_segmentation(segmentation)
    np.save("calibration/sim/seg_mask.npy", segmentation)


def get_seg_mask(image_path, save_path):
    image = Image.open(image_path).convert("RGBA")
    # Separate RGB and Alpha (transparency) channels
    # rgb_image = image.convert("RGB")
    alpha_channel = np.array(image.split()[-1])  # Extract the alpha channel
    # Create a segmentation mask (binary mask: 1 for foreground, 0 for background)
    segmentation_mask = (alpha_channel > 0).astype(
        np.uint8
    )  # 0: background, 1: foreground
    # rgb_array = np.array(rgb_image)
    # Save or visualize results
    Image.fromarray((segmentation_mask * 255).astype(np.uint8)).save(save_path)


def depth_to_point_cloud(rgb, depth, intrinsic):
    """
    Convert depth image to a point cloud.
    Args:
        rgb: RGB image.
        depth: Depth image.
        intrinsic: Camera intrinsic matrix (3x3).
    Returns:
        point_cloud: Nx3 array of 3D points.
        colors: Nx3 array of RGB values.
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth[v, u]
            if z == 0:  # Skip invalid depth values
                continue
            # if z > 0.1:
            #     continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(rgb[v, u] / 255.0)

    return np.array(points), np.array(colors)


def align_point_clouds_with_rgb(
    src_points, src_colors, tgt_points, tgt_colors, visualize=False
):
    """
    Align two point clouds with RGB colors using Open3D's ICP algorithm, with optional visualization.
    Args:
        src_points: Source point cloud (Nx3).
        src_colors: RGB colors for the source point cloud (Nx3).
        tgt_points: Target point cloud (Nx3).
        tgt_colors: RGB colors for the target point cloud (Nx3).
        visualize: Whether to visualize the point clouds before and after alignment.
    Returns:
        transformation: 4x4 transformation matrix.
    """
    # Convert numpy arrays to Open3D point clouds
    src_cloud = o3d.geometry.PointCloud()
    src_cloud.points = o3d.utility.Vector3dVector(src_points)
    src_cloud.colors = o3d.utility.Vector3dVector(src_colors)  # Assign RGB colors

    tgt_cloud = o3d.geometry.PointCloud()
    tgt_cloud.points = o3d.utility.Vector3dVector(tgt_points)
    tgt_cloud.colors = o3d.utility.Vector3dVector(tgt_colors)  # Assign RGB colors

    # Perform ICP alignment
    threshold = 0.0075  # Distance threshold
    reg = o3d.pipelines.registration.registration_icp(
        src_cloud,
        tgt_cloud,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation = reg.transformation

    if visualize:
        # Apply the transformation to the source cloud for visualization
        src_cloud_aligned = src_cloud.transform(transformation)

        # Visualize the point clouds before and after alignment
        print("Visualizing initial point clouds...")
        o3d.visualization.draw_geometries(
            [src_cloud], window_name="Initial Source Point Clouds"
        )
        o3d.visualization.draw_geometries(
            [tgt_cloud], window_name="Initial Target Point Clouds"
        )

        print("Visualizing initial alignment...")
        o3d.visualization.draw_geometries(
            [src_cloud, tgt_cloud], window_name="Initial Alignment"
        )
        print("Visualizing final alignment...")
        o3d.visualization.draw_geometries(
            [src_cloud_aligned, tgt_cloud], window_name="Final Alignment"
        )

    return transformation


def process_sim_observation(raw_obs):
    if isinstance(raw_obs, dict):
        raw_obs = [raw_obs]
    images = {}
    images["2"] = raw_obs[0]["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()
    wrist_img_resolution = (320, 240)
    wrist_img = np.zeros(
        (len(images["2"]), wrist_img_resolution[1], wrist_img_resolution[0], 3)
    )
    for i in range(len(images["2"])):
        wrist_img[i] = cv2.resize(images["2"][i], wrist_img_resolution)
    images["2"] = wrist_img
    return wrist_img[0]


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "guided_dc/cfg/simulation"
    ),  # possibly overwritten by --config-path
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg):
    OmegaConf.resolve(cfg)

    # input(
    #     "Mount the wrist camera to wrist_mount, set pose to identity, and press Enter to continue..."
    # )

    np.set_printoptions(suppress=True, precision=3)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to: {cfg.seed}")
    cfg.env.obs_mode = "rgb+depth+segmentation"
    env = gym.make(cfg.env.env_id, cfg=cfg.env)

    # Step 1: Get a real rgbd image
    real_rgb = cv2.imread("calibration/real/rgb.png")
    real_depth = np.load("calibration/real/depth.npy") * 0.0001  # Unit: meters
    get_seg_mask("calibration/real/seg_rgb.png", "calibration/real/seg_mask.png")
    real_seg_mask = cv2.imread("calibration/real/seg_mask.png")[..., 0] // 255
    # real_seg_mask = np.ones((480, 640))

    real_segmented_rgb = cv2.bitwise_and(
        real_rgb, real_rgb, mask=real_seg_mask.astype(np.uint8)
    )
    real_segmented_depth = real_depth * real_seg_mask
    # Visualize segemented depth image
    plt.imshow(real_segmented_depth * 10)
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("segmented_depth.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Step 2: Get a simulation rgbd image
    get_sim_rgbd(env)
    sim_rgb = cv2.imread("calibration/sim/rgb.png")
    sim_depth = np.load("calibration/sim/depth.npy") / 1000  # Unit: meters
    # get_seg_mask("calibration/sim/seg_rgb.png", "calibration/sim/seg_mask.png")
    # sim_seg_mask = cv2.imread("calibration/sim/seg_mask.png")[..., 0] // 255
    sim_seg_mask = np.load("calibration/sim/seg_mask.npy")
    # sim_seg_mask = np.ones((480, 640))
    sim_segmented_rgb = cv2.bitwise_and(
        sim_rgb, sim_rgb, mask=sim_seg_mask.astype(np.uint8)
    )
    cv2.imwrite("calibration/sim/seg_rgb.png", sim_segmented_rgb)
    sim_segmented_depth = sim_depth * sim_seg_mask

    # Step 3: Use the segmented images to convert rgbd to point cloud and align them

    intrinsic_sim = np.array(cfg.env.camera.wrist_camera.intrinsic)
    intrinsic_real = np.array(
        [
            [391.04746246430125, 0.0, 326.98570796916977],
            [0.0, 391.07204451229177, 242.77958105302346],
            [0.0, 0.0, 1.0],
        ]
    )
    # intrinsic_real = np.array(cfg.env.camera.wrist_camera.intrinsic)

    # Convert to point clouds
    tgt_points, tgt_colors = depth_to_point_cloud(
        real_segmented_rgb, real_segmented_depth, intrinsic_real
    )
    src_points, src_colors = depth_to_point_cloud(
        sim_segmented_rgb, sim_segmented_depth, intrinsic_sim
    )
    # Align point clouds
    transformation = align_point_clouds_with_rgb(
        src_points, src_colors, tgt_points, tgt_colors, visualize=False
    )

    # Step 6: get the transformation matrix and apply it to the wrist camera in simulation
    print("Transformation matrix:\n", transformation)

    # Apply the transformation to the wrist camera in simulation
    # init_pose = sapien_utils.look_at(eye=eye, target=target)
    from scipy.spatial.transform import Rotation as R

    init_q = init_pose.q.cpu().numpy().squeeze()
    init_p = init_pose.p.cpu().numpy().squeeze()
    init_r = R.from_quat([init_q[1], init_q[2], init_q[3], init_q[0]])
    init_rot = init_r.as_matrix()
    init_T = np.eye(4)
    init_T[:3, :3] = init_rot
    init_T[:3, 3] = init_p

    # transformation = np.eye(4)

    current_T = transformation @ init_T
    current_rot = current_T[:3, :3]
    current_pos = current_T[:3, 3]
    current_q = R.from_matrix(current_rot).as_quat()
    current_q = np.array([current_q[3], current_q[0], current_q[1], current_q[2]])

    # current_pos[0] = current_pos[0] + 0.01
    # current_pos[1] = current_pos[1] - 0.005
    # current_pos[2] = current_pos[2] - 0.01

    env.wrist_mount.set_pose(Pose.create_from_pq(current_pos, current_q))
    obs, _ = env.reset()
    wrist_img = process_sim_observation(obs)
    plt.imshow(wrist_img.astype(np.uint8))  # Use cmap='gray' if the image is grayscale
    plt.axis("off")  # Turn off the axes
    plt.gca().set_position(
        [0, 0, 1, 1]
    )  # Remove white space by adjusting the position of the axes

    # Save the image
    plt.savefig("image_2_new.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Calculate relative pose to the panda hand
    panda_hand_pose = env.agent.robot.links_map["panda_hand"].pose
    wrist_mount_pose = env.wrist_mount.pose
    relative_pose = panda_hand_pose.inv() * wrist_mount_pose
    print("Relative pose to the panda hand:\n", relative_pose)

    env.close()


if __name__ == "__main__":
    main()
