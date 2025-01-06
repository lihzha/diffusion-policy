import gc
import gzip
import json
import os
from pathlib import Path
from typing import Union

import cv2
import h5py
import imageio
import numpy as np

try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_rgb_array_videos(video_dir, dir_to_write):
    # List of video file names
    video_files = sorted(
        [
            f
            for f in os.listdir(video_dir)
            if f.endswith(".mp4") and os.path.splitext(f)[0].isdigit()
        ],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # Load video clips
    clips = [VideoFileClip(os.path.join(video_dir, f)) for f in video_files]

    # Concatenate clips
    final_clip = concatenate_videoclips(clips)

    # Write the output to a file
    final_clip.write_videofile(dir_to_write, codec="libx264")

    # After merging, delete the original files
    for file in video_files:
        os.remove(os.path.join(video_dir, file))


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def load_txt(filename: Union[str, Path]):
    return Path(filename).read_text(encoding="utf-8")


def load_hdf5(
    file_path,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = ["observation/timestamp/skip_action"]
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    camera_indices_raw = []
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
                    camera_indices_raw.append(int(cam))
            else:
                output[key] = h5_file[key][()]
        else:
            print(f"Key '{key}' not found in the HDF5 file.")

    # make sure to close h5 file
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            try:
                obj.close()
            except Exception:
                pass
    h5_file.close()

    return output, camera_indices_raw

def load_sim_hdf5_for_training(
    file_path,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = []
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
            else:
                output[key] = h5_file[key][()]
        else:
            print(f"Key '{key}' not found in the HDF5 file.")
    
    if "pick_obj_pos" in h5_file:
        output["pick_obj_pos"] = h5_file["pick_obj_pos"][()]
        output["pick_obj_rot"] = h5_file["pick_obj_rot"][()]
        output["place_obj_pos"] = h5_file["place_obj_pos"][()]
        output["place_obj_rot"] = h5_file["place_obj_rot"][()]

    output["observation/timestamp/skip_action"] = np.zeros(
        len(output["action/gripper_position"])
    ).astype(bool)

    camera_indices_raw = [0, 1, 2]

    # make sure to close h5 file
    # for obj in gc.get_objects():
    #     if isinstance(obj, h5py.File):
    #         try:
    #             obj.close()
    #         except Exception:
    #             pass
    h5_file.close()
    return output, camera_indices_raw


def dict_to_omegaconf_format(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert int32/int64 to Python int
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert float32/float64 to Python float
    elif isinstance(obj, list):
        return [
            dict_to_omegaconf_format(item) for item in obj
        ]  # Recurse for list elements
    elif isinstance(obj, dict):
        return {
            k: dict_to_omegaconf_format(v) for k, v in obj.items()
        }  # Recurse for dict values
    else:
        return obj  # Return unchanged if type is already supported


def save_array_to_video(video_path, img_array, fps=30, brg2rgb=False):
    assert img_array[0].shape[2] == 3, "Image array must be in HWC order"

    writer = imageio.get_writer(video_path, fps=fps, quality=5)
    for im in img_array:
        if brg2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.append_data(im)
    writer.close()


def stack_videos_horizontally(
    video1: np.ndarray,
    video2: np.ndarray,
    output_path: str,
    fps: int = 30,
    bgr2rgb=False,
):
    """
    Stack two videos horizontally and save as a single video file.

    Args:
        video1 (np.ndarray): First video array with shape (length, width, height, channel).
        video2 (np.ndarray): Second video array with shape (length, width, height, channel).
        output_path (str): Path to save the combined video.
        fps (int): Frames per second for the output video.
    """
    # Ensure both videos have the same length
    assert (
        video1.shape[0] == video2.shape[0]
    ), "Both videos must have the same number of frames."

    # Determine the target size for resizing
    target_width = min(video1.shape[2], video2.shape[2])
    target_height = min(video1.shape[1], video2.shape[1])

    def resize_frames(frames: np.ndarray, target_size) -> np.ndarray:
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)
        return np.array(resized_frames)

    # Resize both videos
    resized_video1 = resize_frames(video1, (target_width, target_height))
    resized_video2 = resize_frames(video2, (target_width, target_height))

    # Horizontally stack the frames
    stacked_frames = [
        np.hstack((frame1, frame2))
        for frame1, frame2 in zip(resized_video1, resized_video2)
    ]

    writer = imageio.get_writer(output_path, fps=fps, quality=5)
    images_iter = stacked_frames
    for im in images_iter:
        if bgr2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.append_data(im)
    writer.close()

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # # Write frames to the output video
    # for frame in stacked_frames:
    #     out.write(
    #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     )  # Convert RGB to BGR for OpenCV

    # out.release()


def accelerate_videos(video_paths, coefficient, output_paths):
    """
    Accelerate a list of videos by a given coefficient and save to new paths using OpenCV.

    Parameters:
        video_paths (list of str): List of input video file paths.
        coefficient (float): Speed-up coefficient (e.g., 2 for double speed).
        output_paths (list of str): List of output file paths where the accelerated videos will be saved.

    Returns:
        None
    """
    if len(video_paths) != len(output_paths):
        raise ValueError(
            "The number of video paths must match the number of output paths."
        )

    for input_path, output_path in zip(video_paths, output_paths):
        try:
            # Open the input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_path}")

            # Get the original video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate the new FPS
            new_fps = fps * coefficient

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write every nth frame based on the coefficient
                if frame_index % int(coefficient) == 0:
                    out.write(frame)

                frame_index += 1

            # Release resources
            cap.release()
            out.release()

        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


if __name__ == "__main__":
    video_paths = [f"videos/1056700/200/{i}/False.mp4" for i in range(21, 31)]
    output_paths = [f"output{i}.mp4" for i in range(10)]
    coefficient = 3

    accelerate_videos(video_paths, coefficient, output_paths)
