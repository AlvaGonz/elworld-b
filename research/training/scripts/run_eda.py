import numpy as np
import cv2
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

def run_eda(data_dir, output_dir):
    print(f"Running EDA on: {data_dir}")
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not npz_files:
        print("No .npz files found.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_actions = []
    
    print("Loading data...")
    for f in npz_files:
        data = np.load(os.path.join(data_dir, f))
        all_obs.append(data['obs'][:1000]) # Sample for speed but enough for variance
        all_actions.append(data['act'])
        data.close()
    
    obs = np.concatenate(all_obs, axis=0) 
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"Analyzing {len(obs)} samples for vision, {len(actions)} samples for actions...")

    # 1. Vision Analysis: Variance Heatmap
    variance = np.var(obs.astype(np.float32), axis=0) 
    avg_variance = np.mean(variance, axis=2) 

    v_min, v_max = avg_variance.min(), avg_variance.max()
    normalized = (avg_variance - v_min) / (v_max - v_min + 1e-5)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
    
    heatmap_path = output_dir / "variance_heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap)
    print(f"Variance heatmap saved to: {heatmap_path}")

    # 2. Action Analysis: Distribution
    # Assuming actions are discrete/multiselect
    action_sums = np.sum(actions, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(action_sums)), action_sums)
    plt.title("Action Distribution")
    plt.xlabel("Action Index")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    dist_path = output_dir / "action_distribution.png"
    plt.savefig(str(dist_path))
    plt.close()
    print(f"Action distribution plot saved to: {dist_path}")

    # 3. Summary Stats
    stats = {
        "num_files": len(npz_files),
        "total_frames": len(actions),
        "obs_shape": obs.shape[1:],
        "action_dim": actions.shape[1],
        "static_pixel_ratio": float(np.sum(avg_variance < (v_max * 0.1)) / avg_variance.size),
        "action_counts": action_sums.tolist()
    }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to: {stats_path}")

if __name__ == "__main__":
    recorded_dir = r"c:\Projects\elsworld\research\training\recorded"
    output_dir = r"c:\Projects\elsworld\research\training\outputs\eda"
    run_eda(recorded_dir, output_dir)
