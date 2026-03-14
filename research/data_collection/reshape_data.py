import numpy as np
import cv2
from pathlib import Path

def reshape_all_data(input_dir, output_dir, target_size=(256, 192)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob("*.npz"))
    print(f"Reshaping {len(files)} files to {target_size}...")
    
    for f in files:
        data = np.load(f)
        obs = data['obs']
        act = data['act']
        
        # Only resize if needed
        if obs.shape[1:3] == (target_size[1], target_size[0]):
            print(f"Skipping {f.name} (already {target_size})")
            continue
            
        print(f"Reshaping {f.name} (from {obs.shape[2]}x{obs.shape[1]})...")
        new_obs = []
        for frame in obs:
            new_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            new_obs.append(new_frame)
            
        new_obs = np.array(new_obs, dtype=np.uint8)
        np.savez_compressed(output_path / f.name, obs=new_obs, act=act)
        print(f"  -> Saved to {output_path / f.name}")

if __name__ == "__main__":
    reshape_all_data("c:/Projects/elsworld/data_collection", "c:/Projects/elsworld/research/training/recorded")
