import argparse
import torch
from elworld.train.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="vision", choices=["vision", "memory", "control", "dreaming", "demo_vision", "demo_memory"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mode = args.mode
    config_path = args.config
    device = args.device

    # Only used in demo_vision and demo_memory modes:
    video_config = {
        "data_file":   None,   # None = first gameplay file
        "output_file": "vision_comparison.mp4",
        "max_frames":  None,   # None = all frames
    }
    # ==========================================

    trainer = Trainer(config_path=config_path, mode=mode, device=device)

    if mode == "demo_vision":
        trainer.extract_vision_video(
            data_file=video_config["data_file"],
            output_file=video_config["output_file"],
            max_frames=video_config["max_frames"],
        )
    elif mode == "demo_memory":
        trainer.extract_memory_video(
            output_file="memory_comparison.mp4",
            max_frames=video_config["max_frames"],
        )
    else:
        trainer.run()