# À propos du code

## Core

### Callback

Un **Callback** est un système de hooks qui permet d'injecter une logique personnalisée dans la boucle d'entraînement
sans modifier le `Trainer` lui-même.

À chaque étape de l'entraînement — avant le démarrage, avant et après chaque époque, et à la fin de l'exécution —
le `Trainer` appelle la méthode correspondante sur chaque callback enregistré.
Cela permet de garder la boucle d'entraînement minimale et sans opinion tout en autorisant des effets de bord
arbitrairement complexes : journalisation des métriques dans TensorBoard, sauvegarde des top-k checkpoints,
arrêt anticipé lorsque la perte de validation stagne, etc.

**Callbacks intégrés**

| Callback | Déclencheur | Effet |
|---|---|---|
| `TensorBoardCallback` | `on_epoch_end` | Écrit les scalaires de perte dans un fichier d'événements TensorBoard |
| `CheckpointCallback` | `on_epoch_end` | Sauvegarde l'état du modèle et de l'optimiseur si la perte est dans le top-k |
| `EarlyStoppingCallback` | `on_epoch_end` | Émet un signal d'arrêt si la perte de validation ne s'améliore pas pendant `patience` époques |

**Ajouter un callback personnalisé**

Hériter de `Callback` et ne surcharger que les méthodes nécessaires :

```python
from src.core.callback.base_callback import Callback

class PrintLossCallback(Callback):
    def on_epoch_end(self, trainer, epoch: int) -> None:
        print(f"[{epoch}] loss = {trainer.last_loss:.6f}")
```

Puis l'enregistrer avant d'appeler `.fit()` :

```python
trainer.callbacks.append(PrintLossCallback())
```

::: core.callback.base_callback.Callback

---

### Evaluator

Un **Evaluator** mesure les performances du modèle à la fin de chaque époque,
indépendamment de la perte d'entraînement.

Là où la perte pilote les mises à jour de gradient, les evaluators sont des observateurs en lecture seule —
ils s'exécutent sous `torch.no_grad()` et rapportent des métriques sans interférer avec l'optimisation.
Chaque evaluator reçoit l'instance complète du `Trainer`, lui donnant accès au modèle,
au device et à tout état nécessaire au calcul de sa métrique.

**Evaluators intégrés**

| Evaluator | Métrique | Source de données |
|---|---|---|
| `MSEEvaluator` | Erreur quadratique moyenne sur données de validation | `DataLoader` de validation |
| `ODEEvaluator` | Résidu ODE sur la grille de collocation | Points de collocation `t` |

**Ajouter un evaluator personnalisé**

Hériter de `Evaluator` et implémenter `run()` :

```python
from src.core.evaluator.base_evaluator import Evaluator

class MaxErrorEvaluator(Evaluator):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def run(self, trainer) -> dict:
        max_err = 0.0
        for batch in self.dataloader.get_val_loader():
            x = batch["x"].to(trainer.device)
            y = batch["y"].to(trainer.device)
            with torch.no_grad():
                err = (trainer.model(x) - y).abs().max().item()
            max_err = max(max_err, err)
        return {"max_error": max_err}
```

Puis l'enregistrer avant d'appeler `.fit()` :

```python
trainer.evaluators.append(MaxErrorEvaluator(make_dataloader(data_config)))
```

**Différence avec les Callbacks**

| | Callback | Evaluator |
|---|---|---|
| Rôle | Effets de bord (journalisation, sauvegarde, arrêt) | Calcul de métriques |
| Valeur de retour | `None` | `dict[str, float]` |
| Contexte de gradient | Sans restriction | Toujours `no_grad()` |

::: core.evaluator.base_evaluator.Evaluator

---

### Bootstrap

Le **bootstrap** est une routine de démarrage unique qui rend tous les composants enregistrés
disponibles dans le :data:`~src.core.registry.REGISTRY` avant l'exécution de tout code d'entraînement.

**Pourquoi est-il nécessaire**

Python n'exécute le code de niveau module — y compris les décorateurs `@REGISTRY.xxx.register(...)` —
que lorsque ce module est **importé**. Si un module n'est jamais importé, ses composants ne sont jamais
enregistrés, et tout appel à `REGISTRY.xxx.get()` lèvera une `KeyError` à l'exécution.

`bootstrap_registry()` résout ce problème en important tous les modules enregistrables en un seul endroit,
garantissant que le registre est entièrement peuplé avant que `build_trainer()` ou `make_dataloader()` soient appelés.

**Sans bootstrap**

```python
REGISTRY.models.get("basic_pinn")
# KeyError: [models] Unknown key: 'basic_pinn'. Available: []
```

**Avec bootstrap**

```python
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()

REGISTRY.models.get("basic_pinn")
# <class 'src.repositories.models.model_PINN.PINN'>
```

**Quand l'appeler**

Appeler `bootstrap_registry()` une seule fois, tout en haut du point d'entrée, avant toute
consultation du registre :

```python
# main.py
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()

from src.core.factory import build_trainer, make_dataloader
...
```

**Ajouter un nouveau composant**

Lors de la création d'une nouvelle ODE, d'un modèle, d'une perte ou d'un dataloader, l'enregistrer en deux étapes :

1. Décorer la classe :
```python
# src/repositories/odes/ode_repository/ode_my_ode.py
@REGISTRY.odes.register("my_ode")
class MyODE:
    ...
```

2. Ajouter l'import dans `bootstrap.py` :
```python
import src.repositories.odes.ode_repository.ode_my_ode  # noqa: F401
```

Le `# noqa: F401` supprime l'avertissement du linter *"importé mais non utilisé"* — l'import
est intentionnel, son seul but étant de déclencher le décorateur comme effet de bord.

::: core.bootstrap

---

### Registry

Le **Registry** est un localisateur de services léger qui associe des clés textuelles à des classes,
permettant de remplacer n'importe quel composant — ODE, modèle, perte, dataloader — depuis un fichier
de configuration sans modifier une seule ligne de code d'entraînement.

**Architecture**

Le :data:`~src.core.registry.REGISTRY` global est une instance de `GlobalRegistry`
exposant quatre sous-registres indépendants :

```
REGISTRY
├── odes          → Classes ODE          (ex. "lotka_voltera")
├── models        → Classes de modèles   (ex. "basic_pinn")
├── losses        → Classes de pertes    (ex. "PINN_LOSS")
└── data_loaders  → Classes DataLoader   (ex. "parquet")
```

Chaque sous-registre est une instance de `Registry` qui stocke un mapping `{ nom → classe }`
et valide les clés aussi bien à l'enregistrement qu'à la consultation.

**Enregistrer un composant**

Utiliser le décorateur `@REGISTRY.xxx.register("clé")` sur la définition de la classe :

```python
from src.core.registry import REGISTRY

@REGISTRY.models.register("basic_pinn")
class PINN(torch.nn.Module):
    ...
```

Des métadonnées optionnelles peuvent être attachées pour la documentation ou l'introspection :

```python
@REGISTRY.models.register("basic_pinn", description="PINN standard", version="1.0")
class PINN(torch.nn.Module):
    ...
```

**Récupérer et instancier**

Récupérer une classe avec `.get()` et l'instancier manuellement :

```python
cls = REGISTRY.models.get("basic_pinn")
model = cls(output_dim=2)
```

Ou instancier directement avec `.build()` :

```python
model = REGISTRY.models.build("basic_pinn", output_dim=2)
```

**Introspection**

```python
# Lister toutes les clés enregistrées
REGISTRY.models.list()
# ['basic_pinn']

# Vérifier si une clé existe
"basic_pinn" in REGISTRY.models
# True

# Métadonnées complètes de toutes les entrées
REGISTRY.models.info()
# {'basic_pinn': {'class': 'PINN', 'description': 'PINN standard', 'version': '1.0'}}

# Vue d'ensemble du registre complet
print(REGISTRY)
# GlobalRegistry:
#   odes: ['lotka_voltera', 'van_der_pol']
#   models: ['basic_pinn']
#   losses: ['PINN_LOSS']
#   data_loaders: ['parquet']
```

**Gestion des erreurs**

L'enregistrement d'une clé en double lève une exception immédiatement :

```python
@REGISTRY.models.register("basic_pinn")  # déjà pris
class AnotherModel:
    ...
# ValueError: [models] 'basic_pinn' is already registered.
# Existing: PINN, New: AnotherModel
```

La consultation d'une clé inconnue lève une exception avec les options disponibles :

```python
REGISTRY.models.get("unknown")
# KeyError: [models] Unknown key: 'unknown'. Available: ['basic_pinn']
```

::: core.registry.Registry
::: core.registry.GlobalRegistry

---

### Schemas

Les **Schemas** sont des modèles [Pydantic](https://docs.pydantic.dev) qui définissent, valident et
sérialisent tous les objets de configuration de l'expérience. Ils constituent la source de vérité unique
pour tous les hyperparamètres — de la physique ODE au calendrier d'entraînement — et peuvent être exportés
en JSON pour une reproductibilité complète des expériences.

**Vue d'ensemble**

```
ExperimentConfig
├── ODESConfig        → Physique ODE & grille de simulation
├── DataConfig        → Construction du DataLoader
├── PhysicsWeights    → Poids des termes de perte
└── TrainingConfig    → Hyperparamètres d'entraînement
```

**Validation**

Tous les champs sont validés à l'instanciation. Les valeurs invalides lèvent une `ValidationError`
immédiatement, avant l'exécution de tout code d'entraînement :

```python
DataConfig(
    parquet_path=Path("data/lotka_volterra.parquet"),
    input_cols=["t"],
    target_cols=["prey", "predator"],
    batch_size=-1,  # ValidationError: batch_size must be > 0
)
```

`ODESConfig` vérifie également que `initial_conditions` correspond à `dimension` :

```python
ODESConfig(
    ode_name=AvailablesODE.LOTKA_VOLTERA,
    parameters={"alpha": 0.67, "beta": 1.33, "delta": 1.0, "gamma": 1.0},
    dimension=2,
    initial_conditions=[1.0],  # ValidationError: got 1 element, expected 2
)
```

**Sérialisation**

N'importe quel schema — ou le `ExperimentConfig` de niveau supérieur — peut être exporté en JSON
et rechargé à l'identique, garantissant la reproductibilité complète des expériences :

```python
# Exporter vers un fichier
path = Path("runs/Lotka-Voltera/config.json")
path.write_text(experiment.model_dump_json(indent=4))

# Recharger depuis un fichier
experiment = ExperimentConfig.model_validate_json(path.read_text())
```

Le `CheckpointManager` appelle `save_config()` automatiquement au démarrage de chaque exécution,
de sorte que chaque dossier de checkpoint contient son propre `config.json`.

**Exemple complet**

```python
from src.core.schemas import (
    ExperimentConfig,
    ODESConfig,
    DataConfig,
    PhysicsWeights,
    TrainingConfig,
    AvailablesODE,
)

experiment = ExperimentConfig(
    ode=ODESConfig(
        ode_name=AvailablesODE.LOTKA_VOLTERA,
        parameters={"alpha": 0.67, "beta": 1.33, "delta": 1.0, "gamma": 1.0},
        t_span=(0.0, 50.0),
        grid_size=2000,
        dimension=2,
        initial_conditions=[1.0, 1.0],
    ),
    data=DataConfig(
        parquet_path=Path("data/lotka_volterra.parquet"),
        input_cols=["t"],
        target_cols=["prey", "predator"],
        batch_size=64,
    ),
    physics=PhysicsWeights(lambda_ode=1.0, lambda_data=1.0),
    training=TrainingConfig(epochs=2000, lr=1e-3),
)
```

::: core.schemas.ODESConfig
::: core.schemas.DataConfig
::: core.schemas.PhysicsWeights
::: core.schemas.TrainingConfig
::: core.schemas.ExperimentConfig