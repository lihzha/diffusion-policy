import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation

from guided_dc.utils.pose_utils import quaternion_to_rotation_matrix


def visualize_trajectory_as_video(
    ee_pose_path, video_filename="ee_trajectory.mp4", fps=30
):
    """
    Visualizes a (traj_len, 7) end effector pose trajectory as a 3D animation and saves it as a video.

    Args:
        ee_pose_path (torch.Tensor): End effector trajectory of shape (traj_len, 7) in the form of position + quaternion.
        video_filename (str): The name of the output video file.
        fps (int): Frames per second for the video.
    """
    # Convert tensor to CPU if it's on the GPU
    if ee_pose_path.is_cuda:
        ee_pose_path = ee_pose_path.cpu()

    # Extract positions and quaternions
    traj_len = ee_pose_path.shape[0]
    positions = ee_pose_path[:, :3].numpy()  # (traj_len, 3)
    quaternions = ee_pose_path[:, 3:].numpy()  # (traj_len, 4)

    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set up plot limits
    ax.set_xlim([positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1])
    ax.set_ylim([positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1])
    ax.set_zlim([positions[:, 2].min() - 0.1, positions[:, 2].max() + 0.1])

    # Initialize a point and an arrow for the end effector
    (point,) = ax.plot([], [], [], "bo", markersize=8)  # Blue dot for position
    arrow = ax.quiver(
        0, 0, 0, 0, 0, 0, color="r"
    )  # Red arrow for orientation (initialize)

    def update_frame(i):
        """
        Update the frame of the animation.

        Args:
            i (int): Frame index.
        """
        # Update point position
        point.set_data(
            [positions[i, 0]], [positions[i, 1]]
        )  # Wrap x, y positions in a list
        point.set_3d_properties([positions[i, 2]])  # Wrap z position in a list

        # Update arrow orientation using quaternion
        R = quaternion_to_rotation_matrix(
            torch.tensor(quaternions[i])
        )  # Rotation matrix
        arrow_data = R @ torch.tensor(
            [1, 0, 0], dtype=torch.float32
        )  # Rotate the x-axis to get the arrow
        # Update arrow with new direction
        arrow.set_segments(
            [
                [
                    [positions[i, 0], positions[i, 1], positions[i, 2]],
                    [
                        positions[i, 0] + arrow_data[0].item(),
                        positions[i, 1] + arrow_data[1].item(),
                        positions[i, 2] + arrow_data[2].item(),
                    ],
                ]
            ]
        )

        return point, arrow

    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=traj_len, interval=1000 / fps)

    # Save the animation as a video file
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="Lihan Zha"), bitrate=1800)
    ani.save(video_filename, writer=writer)
    print(f"Saved trajectory animation to {video_filename}")

    # Close the plot after saving
    plt.close(fig)


def trimesh_to_open3d(trimesh_mesh):
    # Create an empty open3d triangle mesh
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Set vertices and triangles
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

    # Optionally set vertex colors if available
    if trimesh_mesh.visual.vertex_colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            trimesh_mesh.visual.vertex_colors[:, :3] / 255.0
        )

    return o3d_mesh


# Convert trimesh mesh to open3d mesh
def visualize_trimesh(trimesh_mesh, extra_pts=None):
    o3d_mesh = trimesh_to_open3d(trimesh_mesh)
    if extra_pts:
        xyz_pts, normals = extra_pts
        extra_pcd = o3d.geometry.PointCloud()
        extra_pcd.points = o3d.utility.Vector3dVector(xyz_pts)
        extra_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_pts))
        lines = []
        line_colors = []

        # For each point, create a line between the point and the tip of the normal
        scale = 0.1  # Length of the normal visualization (can be adjusted)
        for i in range(len(xyz_pts)):
            # Line from the point to the normal tip
            lines.append([i, len(xyz_pts) + i])
            line_colors.append([0, 0, 0])  # Black lines for the normals (optional)

        # Combine points and normals as points for the LineSet
        all_points = np.vstack([xyz_pts, xyz_pts + normals * scale])

        # Create the LineSet for normal visualization
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(all_points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        o3d.visualization.draw_geometries([o3d_mesh, extra_pcd, line_set])
    else:
        o3d.visualization.draw_geometries([o3d_mesh])
