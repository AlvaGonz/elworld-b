import json
import yaml
from pathlib import Path

def prepare_full_training():
    script_dir = Path(__file__).parent.absolute()
    optuna_dir = script_dir / "outputs" / "optuna"
    best_memory_path = optuna_dir / "best_memory_params.json"
    config_path = script_dir / "config.yaml"

    if not best_memory_path.exists():
        print(f"[WAIT] {best_memory_path} not found yet. Optimization still in progress.")
        return False

    # Load best params
    with open(best_memory_path, 'r') as f:
        best_params = json.load(f)
    print(f"[INFO] Loaded best memory params: {best_params}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update memory config
    config['memory_config']['learning_rate'] = best_params['learning_rate']
    config['memory_config']['hidden_dim'] = best_params['hidden_dim']
    config['memory_config']['num_epochs'] = 20  # Set to full training target
    
    # Ensure absolute paths for checkpoints and data (already set but good to confirm)
    config['general_config']['data_path'] = "/home/viethq/Projects/elworld/research/training/recorded/"
    config['dreaming_config']['vision_checkpoint'] = "/home/viethq/Projects/elworld/research/training/src/checkpoints/vision/best_model"

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[OK] config.yaml updated for Full Training (20 epochs).")

if __name__ == "__main__":
    prepare_full_training()
