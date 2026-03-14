import torch
import tqdm
import time
import json
from pathlib import Path
from datetime import datetime

from elworld.model.vision import VisionModel

class VisionTrainer:
    def __init__(self, vision_config, device='cuda', checkpoint_dir='checkpoints/vision', use_amp=True):
        """
        Args:
            vision_config: Dictionary from config.yaml's vision_config section
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints (default: checkpoints/vision)
            use_amp: Use Automatic Mixed Precision for faster training
        """
        self.vision_config = vision_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Create best_model directory
        self.best_model_dir = self.checkpoint_dir / "best_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = VisionModel(
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
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=vision_config['learning_rate']
        )
        
        # GradScaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_l1 = torch.nn.L1Loss()
        
        self.num_epochs = vision_config['num_epochs']
        self.batch_size = vision_config['batch_size']
        
        # Loss weights
        self.mse_weight = 1.0
        self.l1_weight = 0.5
        self.ssim_weight = 0.5 # We'll compute a simple SSIM-like structural loss if possible
        
        # Track best model
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 0
        
        # Try to resume from existing checkpoint
        self._load_existing_checkpoint()
        
        print(f"\n{'='*60}")
        print("VisionTrainer Initialized (High-Performance)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"AMP (Mixed Precision): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {vision_config['learning_rate']}")
        print(f"Total Epochs: {self.num_epochs}")
        print(f"Loss Weights: MSE={self.mse_weight}, L1={self.l1_weight}")
        print(f"{'='*60}\n")

    def _load_existing_checkpoint(self):
        """Find and load the latest checkpoint if exists."""
        # First, load best model info if exists
        if self.best_model_dir.exists():
            metadata_path = self.best_model_dir / "training_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.best_loss = metadata.get('best_loss', float('inf'))
                self.best_epoch = metadata.get('best_epoch', 0)
        
        # Find latest checkpoint
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            print(f"\n{'='*60}")
            print(f"Found existing checkpoint: {latest_checkpoint.name}")
            print(f"{'='*60}")
            self.load_checkpoint(latest_checkpoint)
            
            # Load epoch number from metadata
            metadata_path = latest_checkpoint / "training_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.start_epoch = metadata.get('epoch', 0)
            print(f"{'-'*60}\n")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint folder."""
        checkpoint_folders = list(self.checkpoint_dir.glob("vision_model_checkpoint_*"))
        if not checkpoint_folders:
            return None
        
        # Extract epoch numbers and find max
        epochs = []
        for folder in checkpoint_folders:
            try:
                epoch_num = int(folder.name.split('_')[-1])
                epochs.append((epoch_num, folder))
            except ValueError:
                continue
        
        if not epochs:
            return None
        
        # Return folder with highest epoch number
        latest = max(epochs, key=lambda x: x[0])
        return latest[1]
    
    def train(self, dataloader):
        self.model.train()
        
        print(f"\n{'='*60}")
        print("Starting Training (Optimized VAE)")
        print(f"{'='*60}")
        print(f"Dataset size: {len(dataloader.dataset):,} samples")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Samples per batch: {self.batch_size}")
        print(f"Total iterations: {self.num_epochs * len(dataloader):,}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Saving: Every epoch + best model")
        if self.start_epoch > 0:
            print(f"Resuming from epoch: {self.start_epoch}")
            print(f"Best loss so far: {self.best_loss:.6f} (epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_vq_loss = 0.0
            
            print(f"\n{'-'*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'-'*60}")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU Memory (start): {mem_before:.2f} MB")
            
            batch_times = []
            
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Training")):
                batch_start = time.time()
                
                if isinstance(batch, dict):
                    inputs = batch['observation'].to(self.device, non_blocking=True)
                else:
                    inputs = batch.to(self.device, non_blocking=True)
                
                # Normalize if needed (data is already in [0, 255] uint8 range)
                if inputs.dtype == torch.uint8 or inputs.max() > 1.0:
                    inputs = inputs.float() / 255.0
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Use AMP for faster training
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        recon_mse = self.criterion_mse(outputs['x_recon'], inputs)
                        recon_l1 = self.criterion_l1(outputs['x_recon'], inputs)
                        
                        recon_loss = self.mse_weight * recon_mse + self.l1_weight * recon_l1
                        vq_loss = outputs['vq_loss']
                        loss = recon_loss + vq_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    recon_mse = self.criterion_mse(outputs['x_recon'], inputs)
                    recon_l1 = self.criterion_l1(outputs['x_recon'], inputs)
                    
                    recon_loss = self.mse_weight * recon_mse + self.l1_weight * recon_l1
                    vq_loss = outputs['vq_loss']
                    loss = recon_loss + vq_loss
                    
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_vq_loss += vq_loss.item()
                
                batch_times.append(time.time() - batch_start)
            
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon_loss / len(dataloader)
            avg_vq = epoch_vq_loss / len(dataloader)
            
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'-'*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary")
            print(f"{'-'*60}")
            print(f"Loss:        {avg_loss:.6f}")
            print(f"  Recon:     {avg_recon:.6f}")
            print(f"  VQ:        {avg_vq:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time:        {epoch_time:.2f}s (avg batch: {avg_batch_time*1000:.2f}ms)")
            
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU Memory:  {mem_after:.2f} MB (peak: {mem_peak:.2f} MB)")
            
            # Update learning rate scheduler
            self.scheduler.step(avg_loss)
            
            # Clear CUDA cache to free unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save checkpoint for current epoch
            checkpoint_dir = self.checkpoint_dir / f"vision_model_checkpoint_{epoch+1}"
            self.save_checkpoint_folder(checkpoint_dir, epoch + 1, avg_loss, avg_recon, avg_vq, current_lr, epoch_time)
            print(f"[OK] Checkpoint saved: {checkpoint_dir.name}/")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint_folder(self.best_model_dir, epoch + 1, avg_loss, avg_recon, avg_vq, current_lr, epoch_time, is_best=True)
                print(f"[OK] New best model saved! Loss: {avg_loss:.6f}")
            
            print(f"{'-'*60}")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"  Best loss: {self.best_loss:.4f} (epoch {self.best_epoch})")
        print(f"  Best model: {self.best_model_dir}/")
        print(f"  Total checkpoints: {self.num_epochs}")
        print(f"{'='*60}")
    
    def save_checkpoint_folder(self, folder_path, epoch, loss, recon_loss, vq_loss, lr, epoch_time, is_best=False):
        """Save model checkpoint in HuggingFace-style folder structure."""
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = folder_path / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer state
        optimizer_path = folder_path / "optimizer.pth"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        scheduler_path = folder_path / "scheduler.pth"
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training metadata
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'config': self.vision_config
        }
        
        metadata_path = folder_path / "training_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model config separately for easy loading
        config_path = folder_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.vision_config, f, indent=2)
        
        # Create README
        readme_path = folder_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Vision Model Checkpoint - Epoch {epoch}\n\n")
            f.write(f"## Training Metrics\n\n")
            f.write(f"- **Epoch:** {epoch}\n")
            f.write(f"- **Total Loss:** {loss:.6f}\n")
            f.write(f"- **Reconstruction Loss:** {recon_loss:.6f}\n")
            f.write(f"- **VQ Loss:** {vq_loss:.6f}\n")
            f.write(f"- **Learning Rate:** {lr:.6f}\n")
            f.write(f"- **Epoch Time:** {epoch_time:.2f}s\n")
            if is_best:
                f.write(f"\n**🏆 This is the best model so far!**\n")
            f.write(f"\n## Files\n\n")
            f.write(f"- `model.pth` - Model weights\n")
            f.write(f"- `optimizer.pth` - Optimizer state\n")
            f.write(f"- `scheduler.pth` - LR scheduler state\n")
            f.write(f"- `training_info.json` - Complete training metadata\n")
            f.write(f"- `config.json` - Model configuration\n")
            f.write(f"\n## Usage\n\n")
            f.write(f"```python\n")
            f.write(f"# Load model\n")
            f.write(f"model = VisionModel(**config)\n")
            f.write(f"model.load_state_dict(torch.load('model.pth'))\n")
            f.write(f"```\n")
    
    def load_checkpoint(self, folder_path):
        """Load model checkpoint from folder."""
        folder_path = Path(folder_path)
        
        # Load model weights
        model_path = folder_path / "model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer state
        optimizer_path = folder_path / "optimizer.pth"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler state
        scheduler_path = folder_path / "scheduler.pth"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load metadata
        metadata_path = folder_path / "training_info.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.best_loss = metadata.get('best_loss', float('inf'))
            self.best_epoch = metadata.get('best_epoch', 0)
            
            print(f"Checkpoint loaded from {folder_path}")
            print(f"  Epoch: {metadata.get('epoch', 'unknown')}")
            print(f"  Loss: {metadata.get('loss', 'unknown')}")
            print(f"  Best Loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
        else:
            print(f"Checkpoint loaded from {folder_path}")
            print(f"  Warning: No metadata found")