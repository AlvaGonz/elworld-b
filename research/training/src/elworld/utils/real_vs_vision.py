import numpy as np
import cv2
import torch
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from elworld.model.vision import VisionModel
from utils import load_config, get_vision_config


def load_vision_model(checkpoint_path, device='cuda'):
    """
    Load trained vision model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint folder (e.g., 'checkpoints/best_model')
        device: Device to run inference on
        
    Returns:
        Loaded VisionModel
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    import json
    with open(config_path, 'r') as f:
        vision_config = json.load(f)
    
    # Create model
    model = VisionModel(
        num_hidden=vision_config['num_hidden'],
        res_layer=vision_config['res_layer'],
        res_hidden=vision_config['res_hidden'],
        input_channels=vision_config['input_channels'],
        num_embedding=vision_config['num_embedding'],
        embedding_dim=vision_config['latent_dim'],
        commitment_cost=vision_config['commitment_cost'],
        decay=vision_config.get('decay', 0.99),
        epsilon=vision_config.get('epsilon', 1e-5)
    ).to(device)
    
    # Load weights
    model_path = checkpoint_path / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"[OK] Model loaded from {checkpoint_path}")
    print(f"  Latent dim: {vision_config['latent_dim']}")
    print(f"  Num embeddings: {vision_config['num_embedding']}")
    
    return model, vision_config


def extract_video(
    data_path,
    checkpoint_path,
    output_path="comparison_video.mp4",
    max_frames=None,
    fps=20,
    device='cuda'
):
    """
    Create side-by-side comparison video: left=actual pixels, right=vision model reconstruction.
    
    Args:
        data_path: Path to .npz file containing gameplay data
        checkpoint_path: Path to trained model checkpoint folder
        output_path: Output video file path
        max_frames: Maximum number of frames to process (None = all)
        fps: Video frame rate
        device: Device to run inference on
        
    Example:
        extract_video(
            data_path="../recorded/elsword_gameplay_01.npz",
            checkpoint_path="checkpoints/best_model",
            output_path="vision_comparison.mp4",
            max_frames=1000,
            fps=20
        )
    """
    print(f"\n{'='*60}")
    print("Creating Vision Model Comparison Video")
    print(f"{'='*60}")
    
    # 1. Load trained model
    print("\n[1/4] Loading trained model...")
    model, vision_config = load_vision_model(checkpoint_path, device)
    
    # 2. Load data
    print("\n[2/4] Loading gameplay data...")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    if 'obs' not in data:
        raise ValueError("'obs' key not found in data file")
    
    observations = data['obs']
    num_frames = len(observations) if max_frames is None else min(len(observations), max_frames)
    height, width = observations.shape[1:3]
    
    print(f"  Total frames: {len(observations)}")
    print(f"  Processing: {num_frames} frames")
    print(f"  Frame size: {width}x{height}")
    
    # 3. Setup video writer (side-by-side: width*2)
    print("\n[3/4] Setting up video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    # 4. Process frames
    print("\n[4/4] Processing frames...")
    print("Format: [Left: Actual] | [Right: Vision Model]")
    
    with torch.no_grad():
        for i in range(num_frames):
            # Get actual frame
            actual_frame = observations[i]  # (H, W, C) in RGB
            
            # Preprocess for model: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            model_input = actual_frame.transpose(2, 0, 1)  # (C, H, W)
            model_input = torch.from_numpy(model_input).float().unsqueeze(0).to(device)
            
            # Normalize if needed
            if model_input.max() > 1.0:
                model_input = model_input / 255.0
            
            # Model inference
            outputs = model(model_input)
            reconstructed = outputs['x_recon']
            
            # Convert back to numpy: (1, C, H, W) -> (H, W, C)
            reconstructed = reconstructed.squeeze(0).cpu().numpy()  # (C, H, W)
            reconstructed = reconstructed.transpose(1, 2, 0)  # (H, W, C)
            reconstructed = (reconstructed * 255.0).clip(0, 255).astype(np.uint8)
            
            # Create side-by-side comparison
            # Left: actual, Right: reconstructed
            combined_frame = np.ascontiguousarray(np.concatenate([actual_frame, reconstructed], axis=1))
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_frame, "ACTUAL", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, "VISION MODEL", (width + 10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(combined_frame, f"Frame: {i+1}/{num_frames}", 
                       (width - 100, height - 10), font, 0.5, (255, 255, 255), 1)
            
            # Convert RGB to BGR for OpenCV
            combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(combined_frame_bgr)
            
            # Progress
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Processed: {i+1}/{num_frames} frames ({(i+1)/num_frames*100:.1f}%)")
    
    # Cleanup
    out.release()
    data.close()
    
    print(f"\n{'='*60}")
    print(f"[OK] Video created successfully!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Frames: {num_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width*2}x{height}")
    print(f"{'='*60}\n")