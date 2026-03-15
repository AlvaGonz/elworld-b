"""
Main Trainer: orchestrates VQ-VAE, Memory, Control, and Dreaming modes.

GPU budget: RTX 3050 Laptop 4 GB VRAM
  - Vision  : batch 128 → ~1.5 GB peak
  - Memory  : batch 32  → ~2 GB peak (after encoder is freed)
  - Dreaming: VQ-VAE decoder + MemoryModel → ~1.5 GB peak (no encoder)
"""

import torch
from pathlib import Path

from utils import load_config, get_general_config, get_vision_config, get_memory_config, get_dreaming_config
from elworld.train.pipeline.vision_trainer import VisionTrainer
from elworld.train.pipeline.memory_trainer import MemoryTrainer
from elworld.preprocess.data_setup import setup_data
from elworld.utils.real_vs_vision import extract_video


VALID_MODES = ["vision", "memory", "control", "dreaming", "demo_vision", "demo_memory"]


class Trainer:
    """
    Orchestrates training / evaluation for all model components.

    Modes:
        vision        — Train VQ-VAE (image reconstruction)
        memory        — Train MinGPT (next-frame token prediction)
        control       — Train controller (not yet implemented)
        dreaming      — Run DreamingWorld: live action-conditioned video generation
        demo_vision   — Offline vision model comparison video
        demo_memory   — Offline memory model comparison video
    """

    def __init__(self, config_path: str = "config.yaml", mode: str = "vision", device: str = None):
        self.config_path = config_path
        self.mode = mode.lower()

        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from: {VALID_MODES}")

        self.config = load_config(config_path)
        self.general_config  = get_general_config(self.config)
        self.vision_config   = get_vision_config(self.config)
        self.memory_config   = get_memory_config(self.config)
        self.dreaming_config = get_dreaming_config(self.config)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._print_header()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_header(self):
        print(f"\n{'='*60}", flush=True)
        print(f"  Elworld Trainer  |  mode={self.mode.upper()}  |  device={self.device}", flush=True)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            print(f"  GPU: {props.name}  |  VRAM: {vram_gb:.1f} GB", flush=True)
        print(f"{'='*60}\n", flush=True)

    @staticmethod
    def _section(title: str):
        print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def train_vision(self):
        self._section("Training Vision Model (VQ-VAE)")
        data_path = self.general_config.get("data_path", "../recorded")

        loader = setup_data(
            mode="vision",
            data_path=data_path,
            vision_config=self.vision_config,
            general_config=self.general_config,
        )

        trainer = VisionTrainer(
            vision_config=self.vision_config,
            device=self.device,
            checkpoint_dir="checkpoints/vision",
        )
        trainer.train(loader)

    def train_memory(self):
        self._section("Training Memory Model (MinGPT)")
        data_path = self.general_config.get("data_path", "../recorded")

        loader = setup_data(
            mode="memory",
            data_path=data_path,
            memory_config=self.memory_config,
            vision_config=self.vision_config,
            general_config=self.general_config,
        )

        trainer = MemoryTrainer(
            memory_config=self.memory_config,
            device=self.device,
            checkpoint_dir="checkpoints/memory",
        )
        trainer.train(loader)

    def train_control(self):
        self._section("Training Control Model")
        print("[TODO] Control model not yet implemented.")

    def run_dreaming(self):
        """Launch the interactive DreamingWorld loop."""
        self._section("Dreaming World (Memory Model)")

        from elworld.dreaming.dreaming_world import DreamingWorld

        vision_ckpt  = self.dreaming_config.get("vision_checkpoint",  "checkpoints/vision/best_model")
        memory_ckpt  = self.dreaming_config.get("memory_checkpoint",  "checkpoints/memory/best_memory_model")
        temperature  = self.dreaming_config.get("temperature",  1.0)
        top_k        = self.dreaming_config.get("top_k",        50)
        save_path    = self.dreaming_config.get("save_path",    None)  # None = display only

        world = DreamingWorld(
            memory_checkpoint=memory_ckpt,
            vision_checkpoint=vision_ckpt,
            memory_config=self.memory_config,
            device=self.device,
        )
        world.run(temperature=temperature, top_k=top_k, save_path=save_path)

    def extract_vision_video(self, data_file=None, output_file=None, max_frames=None):
        self._section("Extracting Vision Model Comparison Video")
        data_path = self.general_config.get("data_path", "../recorded")

        if data_file is None:
            data_file = f"{data_path}/elsword_gameplay_01.npz"
        elif not data_file.startswith("/"):
            data_file = f"{data_path}/{data_file}"

        if output_file is None:
            output_file = "vision_comparison.mp4"

        extract_video(
            data_path=data_file,
            checkpoint_path="checkpoints/vision/best_model",
            output_path=output_file,
            max_frames=max_frames,
            fps=self.general_config.get("frame_rate", 20),
            device=str(self.device),
        )

    def extract_memory_video(self, output_file=None, max_frames=None):
        self._section("Extracting Memory Model Comparison Video")
        from elworld.utils.real_vs_memory import evaluate_memory
        data_path = self.general_config.get("data_path", "../recorded")
        
        if output_file is None:
            output_file = "memory_comparison.mp4"
            
        evaluate_memory(
            data_path=data_path,
            vision_checkpoint="checkpoints/vision/best_model",
            memory_checkpoint="checkpoints/memory/best_memory_model",
            memory_config=self.memory_config,
            output_path=output_file,
            max_frames=max_frames if max_frames is not None else 200,
            fps=self.general_config.get("frame_rate", 20),
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def run(self):
        if self.mode == "vision":
            self.train_vision()
        elif self.mode == "memory":
            self.train_memory()
        elif self.mode == "control":
            self.train_control()
        elif self.mode == "dreaming":
            self.run_dreaming()
        elif self.mode == "demo_vision":
            self.extract_vision_video(max_frames=1500)
        elif self.mode == "demo_memory":
            self.extract_memory_video(max_frames=1500)

        print(f"\n{'='*60}\n  Task complete! [OK]\n{'='*60}\n")
