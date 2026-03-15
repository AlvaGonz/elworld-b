import optuna
import torch
import os
import sys
import json
from pathlib import Path
import shutil
import gc

# Add current directory to path for elworld imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elworld.train.pipeline.memory_trainer import MemoryTrainer
from elworld.preprocess.data_setup import setup_data
from utils import load_config, get_general_config, get_vision_config, get_memory_config

# Path setup
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "optuna"
CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints" / "optuna_memory"

# Global data loader to avoid re-encoding every trial
global_loader = None

def get_loader(vision_config, memory_config, general_config):
    global global_loader
    if global_loader is None:
        data_path = general_config.get("data_path", "../recorded")
        # Add vision_checkpoint_path to general_config for data_setup to use
        vision_ckpt = general_config.get("vision_checkpoint_path")
        general_config["vision_checkpoint_path"] = vision_ckpt

        global_loader = setup_data(
            mode="memory",
            data_path=data_path,
            memory_config=memory_config,
            vision_config=vision_config,
            general_config=general_config,
        )
    return global_loader

def objective(trial):
    # Load base config
    config = load_config(str(CONFIG_PATH))
    general_config = get_general_config(config)
    vision_config = get_vision_config(config).copy()
    memory_config = get_memory_config(config).copy()

    # Apply best vision params
    best_vision_params_path = OUTPUT_DIR / "best_vision_params.json"
    if best_vision_params_path.exists():
        with open(best_vision_params_path, "r") as f:
            best_vision_params = json.load(f)
        vision_config.update(best_vision_params)
        memory_config['vocab_size'] = vision_config['num_embedding']
        memory_config['embed_dim'] = vision_config['latent_dim']
    
    # Ensure vision_checkpoint_path is in general_config
    if "vision_checkpoint" in config.get("dreaming_config", {}):
        general_config["vision_checkpoint_path"] = config["dreaming_config"]["vision_checkpoint"]

    # Suggest hyperparameters
    memory_config['learning_rate'] = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)
    memory_config['hidden_dim'] = trial.suggest_categorical('hidden_dim', [256, 512])
    memory_config['num_epochs'] = 5 
    
    # Use a unique checkpoint dir for each trial
    trial_dir = CHECKPOINT_ROOT / f"trial_{trial.number}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = MemoryTrainer(
        memory_config=memory_config,
        device=device,
        checkpoint_dir=str(trial_dir)
    )

    # Get loader (encodes only on first call)
    loader = get_loader(vision_config, memory_config, general_config)

    # Custom training loop
    trainer.model.train()
    best_loss = float('inf')

    for epoch in range(memory_config['num_epochs']):
        epoch_loss = 0.0
        
        for batch in loader:
            input_tokens = batch['input_tokens'].to(device, non_blocking=True)
            actions = batch['actions'].to(device, non_blocking=True)
            target_tokens = batch['target_tokens'].to(device, non_blocking=True)
            
            trainer.optimizer.zero_grad(set_to_none=True)
            
            if trainer.use_amp:
                with torch.amp.autocast('cuda'):
                    output = trainer.model(input_tokens, actions=actions, targets=target_tokens)
                    loss = output['loss']
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                output = trainer.model(input_tokens, actions=actions, targets=target_tokens)
                loss = output['loss']
                loss.backward()
                trainer.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"  [Trial {trial.number}] Epoch {epoch+1}/{memory_config['num_epochs']} - Loss: {avg_loss:.6f}")
        trial.report(avg_loss, epoch)

        if trial.should_prune():
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Clean up trial (but KEEP loader)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_loss

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=10)

    print("Best memory trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    
    with open(OUTPUT_DIR / "best_memory_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
