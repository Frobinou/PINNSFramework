# PINN Trainer

Entraîneur de réseau de neurones informé par la physique (*Physics-Informed Neural Network*), combinant une perte résiduelle d'ODE avec une perte sur données empiriques.
Construit sur PyTorch avec journalisation TensorBoard et sauvegarde automatique des top-k checkpoints.

---

## Vue d'ensemble

L'entraîneur minimise une perte composite :

$$
\mathcal{L} = \lambda_{\text{ode}} \cdot (\mathcal{L}_{\text{ode}} + \mathcal{L}_{\text{ic}}) + \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}
$$

| Terme | Description |
|---|---|
| $\mathcal{L}_{\text{ode}}$ | Résidu quadratique moyen de l'ODE sur la grille de collocation |
| $\mathcal{L}_{\text{ic}}$ | Erreur quadratique moyenne sur les conditions initiales en $t=0$ |
| $\mathcal{L}_{\text{data}}$ | Erreur quadratique moyenne sur les observations empiriques (Parquet) |

Trois modes d'entraînement sont disponibles selon les valeurs de $\lambda$ :

| Mode | `lambda_ode` | `lambda_data` |
|---|---|---|
| Physique seule | > 0 | 0 |
| Données seules | 0 | > 0 |
| PINN hybride | > 0 | > 0 |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances requises :**

- `torch >= 2.0`
- `numpy`
- `matplotlib`
- `scipy`
- `tensorboard`
- `pyarrow` / `pandas` (pour le chargement des données Parquet)

---

## Structure du projet

```
src/
├── core/
│   ├── checkpoint_manager.py   # Logique de sauvegarde top-k
│   ├── data_loader.py          # Construction du DataLoader Parquet
│   └── models.py               # Dataclass TrainingStepLog
├── data_models.py              # TrainingConfig, DataConfig, ODEExperiment
├── ai_model_repository.py      # Registre des réseaux de neurones
├── odes/                       # Registre et implémentations des ODE
├── logger.py                   # Configuration du logger
└── trainer.py                  # Entraîneur (ce module)
```

---

## Démarrage rapide

```python
from pathlib import Path
from src.trainer import Trainer
from src.data_models import TrainingConfig, ODEExperiment

training_config = TrainingConfig(
    epochs=5000,
    l_r=1e-3,
    log_frequency=100,
    top_k_save_frequency=5,
    model_name="MLP",
)

ode_experiment_config = ODEExperiment(
    ode_config=...,         # Nom de l'ODE, paramètres, grid_size, t_span, lambda_ode
    data_config=None,       # Remplacer par DataConfig(...) pour le mode hybride
    model_dimension=2,
    initial_conditions=[1.0, 0.5],
    input_cols=["t"],
    target_cols=["prey", "predator"],
)

trainer = Trainer(
    training_config=training_config,
    ode_experiment_config=ode_experiment_config,
    output_folder_path=Path("outputs/"),
)

trainer.run()
```

---

## Configuration

### `TrainingConfig`

| Champ | Type | Description |
|---|---|---|
| `epochs` | `int` | Nombre d'époques d'entraînement |
| `l_r` | `float` | Taux d'apprentissage Adam |
| `log_frequency` | `int` | Intervalle d'époques entre deux écritures TensorBoard |
| `top_k_save_frequency` | `int` | Nombre maximum de checkpoints à conserver |
| `model_name` | `str` | Clé dans `AIMODEL_REPOSITORY` |

### `ODEExperiment`

| Champ | Type | Description |
|---|---|---|
| `ode_config` | `ODEConfig \| None` | Spécification de l'ODE ; `None` désactive la perte physique |
| `data_config` | `DataConfig \| None` | Spécification des données ; `None` désactive la perte sur données |
| `model_dimension` | `int` | Nombre de variables d'état de l'ODE |
| `initial_conditions` | `list[float]` | Vecteur d'état initial $x_0$ |
| `input_cols` | `list[str]` | Noms des colonnes d'entrée dans le fichier Parquet |
| `target_cols` | `list[str]` | Noms des colonnes cibles — utilisés aussi comme étiquettes de variables dans TensorBoard |

### `ODEConfig`

| Champ | Type | Description |
|---|---|---|
| `ode_name` | `str` | Clé dans `ODE_REPOSITORY` |
| `parameters` | `dict` | Paramètres spécifiques à l'ODE (ex. `{"alpha": 1.0, "beta": 0.1}`) |
| `grid_size` | `int` | Nombre de points de collocation |
| `t_span` | `tuple[float, float]` | Intervalle d'intégration $[t_0, t_f]$ |
| `lambda_ode` | `float` | Poids de la perte physique |

---

## Structure du répertoire de sortie

Chaque appel à `run()` crée un dossier d'expérience horodaté :

```
outputs/
└── experiment_YYYY-MM-DD_HH-MM-SS/
    ├── tensorboard_logs/       # Sortie du SummaryWriter
    ├── save/                   # Top-k checkpoints (.pt)
    ├── training_config.json
    └── ode_experiment_config.json
```

Lancer TensorBoard avec :

```bash
tensorboard --logdir outputs/
```

---

## Métriques TensorBoard

### Scalaires d'entraînement

| Tag | Description |
|---|---|
| `Training/loss/total` | Perte totale pondérée |
| `Training/loss/physics` | Perte résiduelle ODE (quand `lambda_ode > 0`) |
| `Training/loss/data` | Perte sur données (quand `lambda_data > 0`) |
| `Training/residuals/<var_name>` | Résidu absolu moyen par variable d'état |
| `Training/residuals/mean` | Résidu moyen global sur tous les points et variables |
| `Training/residuals/max` | Résidu absolu maximal — signale les points de collocation mal satisfaits |
| `Training/gradients/global_norm` | Norme L2 globale des gradients — diagnostique les gradients évanescents ou explosifs |

### Scalaires d'évaluation

| Tag | Description |
|---|---|
| `Evaluation/MSE` | MSE globale par rapport à la solution de référence scipy |
| `Evaluation/MSE/<var_name>` | MSE par variable (ex. `Evaluation/MSE/prey`) |

### Images d'évaluation

| Tag | Description |
|---|---|
| `Evaluation/Observables/DynamicTrajectories` | Trajectoires temporelles : prédiction vs référence |
| `Evaluation/Observables/DynamicPhaseTrajectories` | Portrait de l'espace des phases à l'époque courante |
| `Evaluation/Observables/PhasePortraitOverlay` | Superposition des portraits de phase sur les époques — visualise la convergence vers l'attracteur |
| `Evaluation/Residuals/CollocationHeatmap` | Carte de chaleur de `\|résidu\|` sur la grille de collocation ; une ligne par variable, le temps sur l'axe x |

---

## Exemple : Lotka-Volterra

```python
ode_config = ODEConfig(
    ode_name="LotkaVolterra",
    parameters={"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
    grid_size=1000,
    t_span=(0.0, 15.0),
    lambda_ode=1.0,
)

ode_experiment_config = ODEExperiment(
    ode_config=ode_config,
    data_config=None,
    model_dimension=2,
    initial_conditions=[10.0, 5.0],
    input_cols=["t"],
    target_cols=["prey", "predator"],  # utilisés comme var_names dans tous les tags TensorBoard
)
```

Tags TensorBoard attendus pour cette configuration :

- `Training/residuals/prey`, `Training/residuals/predator`
- `Evaluation/MSE/prey`, `Evaluation/MSE/predator`
- `Evaluation/Residuals/CollocationHeatmap` — deux lignes (prey / predator)
- `Evaluation/Observables/PhasePortraitOverlay` — plan de phase prey vs predator

---

## Ajouter une nouvelle ODE

1. Implémenter une classe héritant de l'interface ODE de base :

```python
class MyODE:
    def __init__(self, params: dict): ...

    def torch_ode(self, y: torch.Tensor) -> torch.Tensor:
        """Membre de droite F(y, t) — forme (N, n_vars)."""
        ...

    def simulate(self, t_span, x0, nb_points):
        """Solution de référence scipy."""
        ...

    def log_trajectory_plot(self, t_true, y_true, y_pred) -> np.ndarray | None:
        """Retourner un tableau uint8 CHW ou None pour ignorer la journalisation."""
        ...

    def log_trajectory_phase_space_plot(self, y_true, y_pred) -> np.ndarray | None:
        ...
```

2. L'enregistrer :

```python
ODE_REPOSITORY.register("MyODE", MyODE)
```

---

## Problèmes connus et limitations

- `var_names` doit être défini explicitement sur l'instance du trainer (`trainer.var_names = ["prey", "predator"]`) ou déduit de `target_cols` — pas de câblage automatique pour l'instant.
- `_phase_overlay_history` croît sans limite. Utiliser `deque(maxlen=N)` pour les longues exécutions :

```python
from collections import deque
trainer._phase_overlay_history = deque(maxlen=20)
```

- `physics_loss_residual` utilise `torch.autograd.grad` et **ne doit pas** être appelé dans un contexte `torch.no_grad()`.