import sys
import os
from pathlib import Path

# Thêm đường dẫn src vào sys.path để có thể import
sys.path.append(str(Path(r"c:\Projects\elsworld\research\training\src")))

from elworld.utils.real_vs_vision import extract_video

if __name__ == "__main__":
    # Get recorded data path
    recorded_dir = r"c:\Projects\elsworld\research\training\recorded"
    data_path = os.path.join(recorded_dir, "elsword_gameplay_01.npz")
    if not os.path.exists(data_path):
        # find the first .npz file
        files = [f for f in os.listdir(recorded_dir) if f.endswith('.npz')]
        if files:
            data_path = os.path.join(recorded_dir, files[0])
            
    print(f"Using data: {data_path}")
    extract_video(
        data_path=data_path,
        checkpoint_path="c:/Projects/elsworld/research/training/src/checkpoints/vision/vision_model_checkpoint_31",
        output_path="c:/Projects/elsworld/research/training/src/vae_test_output_new.mp4",
        max_frames=200,
        fps=20
    )
