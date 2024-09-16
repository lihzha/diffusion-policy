import gzip
import json
from typing import Union
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def merge_rgb_array_videos(video_dir):
    # List of video file names
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4') and os.path.splitext(f)[0].isdigit()],
                     key=lambda x: int(os.path.splitext(x)[0]))

    # Load video clips
    clips = [VideoFileClip(os.path.join(video_dir, f)) for f in video_files]

    # Concatenate clips
    final_clip = concatenate_videoclips(clips)

    # Write the output to a file
    final_clip.write_videofile("./videos/output.mp4", codec="libx264")
    
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