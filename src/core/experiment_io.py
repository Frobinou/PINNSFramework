from datetime import datetime
import json
from pathlib import Path
from src.core.schemas import ExperimentConfig

def save_experiment(config: ExperimentConfig, base_dir: str = "experiments") -> Path:
    """
    Sauvegarde une expérience dans un fichier JSON horodaté.
    
    Args:
        config: Configuration complète de l'expérience.
        base_dir: Dossier de destination.
    
    Returns:
        Path du fichier créé.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    output_dir = Path(base_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / 'training_conf.json'
    filepath.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    
    return output_dir

def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))