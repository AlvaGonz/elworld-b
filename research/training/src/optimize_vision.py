import optuna
import torch
import os
import sys
from pathlib import Path
import shutil
import gc

# Add current directory to path for elworld imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elworld.train.pipeline.vision_trainer import VisionTrainer
from elworld.preprocess.data_setup import setup_data
from utils import load_config, get_general_config, get_vision_config

def objective(trial):
    # Load base config
    config_path = "config.yaml"
    config = load_config(config_path)
    general_config = get_general_config(config)
    vision_config = get_vision_config(config).copy()

    # Suggest hyperparameters
    vision_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    vision_config['latent_dim'] = trial.suggest_categorical('latent_dim', [64, 128])
    vision_config['num_epochs'] = 5  # Limit to 5 epochs per trial
    
    # Use a unique checkpoint dir for each trial to avoid conflicts
    trial_dir = Path(f"checkpoints/optuna_vision/trial_{trial.number}")
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = VisionTrainer(
        vision_config=vision_config,
        device=device,
        checkpoint_dir=str(trial_dir)
    )

    # Setup data
    data_path = general_config.get("data_path", "../recorded")
    loader = setup_data(
        mode="vision",
        data_path=data_path,
        vision_config=vision_config,
        general_config=general_config,
    )

    # Custom training loop to report to Optuna and prune
    trainer.model.train()
    best_recon_loss = float('inf')

    for epoch in range(vision_config['num_epochs']):
        epoch_recon_loss = 0.0
        
        for batch in loader:
            if isinstance(batch, dict):
                inputs = batch['observation'].to(device, non_blocking=True)
            else:
                inputs = batch.to(device, non_blocking=True)
            
            if inputs.dtype == torch.uint8 or inputs.max() > 1.0:
                inputs = inputs.float() / 255.0
            
            trainer.optimizer.zero_grad(set_to_none=True)
            
            if trainer.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = trainer.model(inputs)
                    recon_mse = trainer.criterion_mse(outputs['x_recon'], inputs)
                    recon_l1 = trainer.criterion_l1(outputs['x_recon'], inputs)
                    recon_loss = trainer.mse_weight * recon_mse + trainer.l1_weight * recon_l1
                    vq_loss = outputs['vq_loss']
                    loss = recon_loss + vq_loss
                
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                outputs = trainer.model(inputs)
                recon_mse = trainer.criterion_mse(outputs['x_recon'], inputs)
                recon_l1 = trainer.criterion_l1(outputs['x_recon'], inputs)
                recon_loss = trainer.mse_weight * recon_mse + trainer.l1_weight * recon_l1
                vq_loss = outputs['vq_loss']
                loss = recon_loss + vq_loss
                loss.backward()
                trainer.optimizer.step()
            
            epoch_recon_loss += recon_loss.item()
        
        avg_recon = epoch_recon_loss / len(loader)
        trial.report(avg_recon, epoch)

        if trial.should_prune():
            # Clean up before raising
            del trainer
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if avg_recon < best_recon_loss:
            best_recon_loss = avg_recon

    # Clean up trial
    del trainer
    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_recon_loss

if __name__ == "__main__":
    study_dir = Path("outputs/optuna")
    study_dir.mkdir(parents=True, exist_ok=True)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=10) # 10 trials as a start

    print("Best vision trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    
    # Save best params
    import json
    with open(study_dir / "best_vision_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
