import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

from src.core.inference_runner import InferenceRunner

model_path = Path("runs\Lotka-Voltera\experiment_2026-04-17_16-09-47")
runner = InferenceRunner.from_config(experiment_dir=model_path)
runner.load()
runner.plot()