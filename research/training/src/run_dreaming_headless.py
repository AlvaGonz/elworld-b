import torch
import cv2
import numpy as np
import json
from pathlib import Path
from elworld.model.memory import MemoryModel
from elworld.dreaming.dream_renderer import DreamRenderer
from utils import load_config, get_general_config, get_memory_config, get_dreaming_config

def run_headless_dreaming():
    # Setup paths and config
    src_dir = Path(__file__).parent.absolute()
    config_path = src_dir / "config.yaml"
    config = load_config(str(config_path))
    
    gen_cfg = get_general_config(config)
    mem_cfg = get_memory_config(config)
    dream_cfg = get_dreaming_config(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    vision_ckpt = src_dir / "checkpoints/vision/best_model"
    # Use the best memory model we just trained
    memory_ckpt = src_dir / "checkpoints/memory/best_memory_model"
    
    print(f"Loading Models...")
    print(f"  Vision: {vision_ckpt}")
    print(f"  Memory: {memory_ckpt}")

    # Load Memory Model
    memory = MemoryModel(
        vocab_size=mem_cfg.get('vocab_size', 1024),
        context_frames=mem_cfg.get('context_frames', 4),
        embed_dim=mem_cfg.get('embed_dim', 128),
        hidden_dim=mem_cfg.get('hidden_dim', 256),
        num_res_blocks=mem_cfg.get('num_res_blocks', 4),
        action_dim=mem_cfg.get('action_dim', 22)
    ).to(device)

    if mem_cfg.get("use_qat", True):
        memory.prepare_qat()
    
    memory.load_state_dict(torch.load(memory_ckpt / "model.pth", map_location=device))
    memory.eval()

    # Load Renderer
    renderer = DreamRenderer(str(vision_ckpt), device=device)

    # Setup Video Writer
    output_path = src_dir / "outputs" / "dreaming" / "headless_dream.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (256, 192))

    # Initialize Seed (Random for now)
    context_frames = mem_cfg.get('context_frames', 4)
    context = torch.randint(
        0, mem_cfg.get("vocab_size", 1024),
        (1, context_frames, 24, 32),
        dtype=torch.long,
        device=device
    )

    print(f"Starting headless dream generation (1500 frames)...")
    
    # Define a simple script: walk right for 100 frames, then jump for 20 frames, then stay still
    # action vector indices (based on KEYS_TO_LOG):
    # 'f8', 'up', 'down', 'left', 'right', 'z', 'x', ...
    # right is index 4
    # z (jump/attack?) is index 5
    
    for i in range(1500):
        action = torch.zeros((1, 22), device=device)
        if i < 100:
            action[0, 4] = 1.0  # Walk Right
        elif 100 <= i < 120:
            action[0, 5] = 1.0  # Action Z
        
        with torch.no_grad():
            new_tokens = memory.generate(
                start_tokens=context,
                actions=action,
                temperature=1.0,
                top_k=50
            ) # [1, 1, 24, 32]
            
            # Roll context
            context = torch.cat([context[:, 1:], new_tokens], dim=1)
            
            # Render
            tokens_flat = new_tokens[0, 0].flatten()
            frame = renderer.decode(tokens_flat)
            
            # Add some info to frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(frame_bgr)
            
        if i % 100 == 0:
            print(f"  Generated {i}/1500 frames...")

    writer.release()
    print(f"[OK] Headless dream saved to {output_path}")

if __name__ == "__main__":
    run_headless_dreaming()
