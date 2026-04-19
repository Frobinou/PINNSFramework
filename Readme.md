# 🚀 ODEPINNSFramework

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docs](https://img.shields.io/badge/docs-MkDocs-blue.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)
![Framework](https://img.shields.io/badge/PINNs-ODE%20Solver-purple.svg)

---

## 🔬 Overview

**ODEPINNSFramework** is a research-grade Python framework for solving **Ordinary Differential Equations (ODEs)** using **Physics-Informed Neural Networks (PINNs)**.

It embeds physical laws directly into neural network training, enabling data-efficient and physically consistent modeling of dynamical systems.

---

## ✨ Key Features

- 🧠 Physics-Informed Neural Networks (PINNs)
- ⚙️ Modular ODE / model / training architecture
- 📉 Hybrid optimization (physics + data losses)
- 🔁 Automatic differentiation for governing equations
- 📊 TensorBoard logging & experiment tracking
- 💾 Checkpointing with top-k model selection
- 🧪 CLI for training, inference, and data generation

---

## 🏗️ Architecture

The framework is structured around three core components:

- **ODE definitions** → mathematical system specification  
- **Neural models** → function approximators  
- **Trainer** → physics-informed optimization loop  

This separation ensures full extensibility for research experimentation.

---

## ⚙️ Installation

```bash
pip install -e .
```

🚀 Quickstart

```python
from core.trainer import Trainer

trainer = Trainer(
    training_config=...,
    ode_experiment_config=...
)

trainer.run()
```

📚 Documentation

Built with MkDocs

▶️ Run locally
```
mkdocs serve
```
Open: http://127.0.0.1:8000/

🏗️ Build static site

```bash
mkdocs build
```
🚀 Deploy to GitHub Pages
```bash
mkdocs gh-deploy
```

## 🧠 Methodology

The framework trains a neural network to approximate the solution of an Ordinary Differential Equation (ODE) by embedding physical constraints directly into the learning process.

### Physics-Informed Learning

Instead of relying solely on data, the model enforces the governing dynamics of the system:

``  
\[
\frac{dy}{dt} = f(y, t)
\]

This is achieved using automatic differentiation to compute derivatives of the neural network output with respect to time.

---

### Loss Function

The total optimization objective is defined as a weighted combination of two terms:

#### 1. Physics Loss

Ensures that the neural network respects the underlying differential equation:

\[
\mathcal{L}_{phys} = \left\| \frac{dy}{dt} - f(y, t) \right\|^2
\]

#### 2. Data Loss (optional)

Used when observations are available:

\[
\mathcal{L}_{data} = \| y_{pred} - y_{true} \|^2
\]

---

### Total Objective

The final loss function is:

\[
\mathcal{L} = \lambda_{phys} \mathcal{L}_{phys} + \lambda_{data} \mathcal{L}_{data}
\]

Loss = λ_phys * L_phys + λ_data * L_data

where:
- \( \lambda_{phys} \) controls the weight of physical constraints
- \( \lambda_{data} \) controls the influence of observed data

---

### Training Principle

The model is optimized such that:
- It satisfies the ODE dynamics globally over the time domain
- It remains consistent with available data points (if provided)
- It generalizes beyond observed trajectories using physics constraints

---

### Key Insight

This approach allows the model to learn **physically consistent trajectories** even in regimes with:
- sparse data
- noisy observations
- partially observed systems

📁 Project Structure
PINNSFramework/
│
├── src/
│   ├── core/
│   ├── odes/
│   ├── ode_repository/
│   ├── model_repository/
│   ├── visualizers/
│   ├── data_models.py
│   ├── trainer_runner.py
│   ├── inference_runner.py
│   ├── run.py
│   └── logger.py
│
├── runs/
├── docs/
├── mkdocs.yml
└── pyproject.toml
🧪 Training
python src/run.py

Outputs include:

checkpoints
TensorBoard logs
evaluation plots
experiment configs

## 🧾 Command Line Interface (CLI)

The framework provides a command-line interface to streamline training, inference, and dataset generation.

It is designed to allow quick experimentation without modifying the source code.

---

## 🚀 Training

Start a training run using a predefined ODE configuration:

```bash id="train1"
python src/cli/app.py train --ode lotka_volterra
```

🧩 Supported Systems
Lotka–Volterra dynamics
CFAST model
Custom ODE systems

🔭 Roadmap
 Uncertainty quantification
 JAX backend support
 Distributed training
 Benchmark suite for PINNs


📌 Design Philosophy
Modularity → every component is replaceable
Reproducibility → full experiment tracking
Physics consistency → constraints enforced by design


📄 License

MIT License


