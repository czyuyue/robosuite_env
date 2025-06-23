import subprocess
import tempfile
import os
import numpy as np
from PIL import Image
import glob

def read_16bit_avi(avi_path):
    """
    Reads a 16-bit grayscale AVI file using ffmpeg to preserve uint16 format.
    Returns a numpy array of shape (num_frames, H, W) with dtype uint16.
    """
    if not os.path.exists(avi_path):
        raise IOError(f"Cannot find video file: {avi_path}")
    
    # Create temporary directory for extracted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use ffmpeg to extract frames as 16-bit PNG files
        frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
        cmd = [
            "ffmpeg", 
            "-i", avi_path,
            "-pix_fmt", "gray16le",
            "-vsync", "0",
            frame_pattern,
            "-y"  # Overwrite output files
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        
        # Get list of extracted frame files
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        
        if len(frame_files) == 0:
            raise RuntimeError("No frames extracted from video.")
        
        # Load frames into numpy array
        frames = []
        for frame_file in frame_files:
            # Load 16-bit PNG using PIL
            img = Image.open(frame_file)
            # Convert to numpy array and ensure uint16 dtype
            frame = np.array(img, dtype=np.uint16)
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            frames.append(frame)
        
        return np.stack(frames, axis=0)

if __name__ == "__main__":
    avi_path = "/data_new/yueyu/zz/robosuite_env/robosuite/robosuite_data/robosuite/data/collision_data/ep_id0_1750683969/depth.avi"
    # avi_path = "/data_new/yueyu/zz/robosuite_env/robosuite/datasets/collision_data/ep_id0_1750682537/depth.avi"
    frames = read_16bit_avi(avi_path)
    print("Loaded frames shape:", frames.shape)
    print("dtype:", frames.dtype)
    print("min:", frames.min(), "max:", frames.max())
    # Optionally, save the first frame as npy for inspection
    # np.save("first_depth_frame.npy", frames[0])
