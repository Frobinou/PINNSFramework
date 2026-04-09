# README - PINNs Framework
This project implements a PINN (Physics-Informed Neural Networks) framework in PyTorch, enabling the solution of:

Simple ODEs and systems of ODEs
Integro-differential equations
Potentially extendable to simple 1D/2D PDEs

The neural network is trained directly using a loss function based on the governing physical equation, without relying on simulated data.

📂 Project Structure

```
PINNSFramework/
│
├── venv/                  # Python virtual environment
├── runs/                  # TensorBoard logs
├── model.py               # PINN network definition
├── train.py               # Training script
├── utils.py               # Helper functions (optional)
└── README.md              # Project documentation
```

⚡ Prerequisites

Python 3.11 (recommended, from official python.org)
PyTorch with CUDA (GPU optional)
TensorBoard for convergence visualization
Windows or Linux (tested on Windows with GTX 1650)

🚀 Quick Installation


- Create a virtual environment 
```bash
python -m venv venv
venv\Scripts\activate
```

- Upgrade pip
```bash
python -m pip install --upgrade pip
````
- Install main dependencies

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib tensorboard numpy

```
If there is some troubles with tensorboard : 
```bash
pip install tensorboard>=2.20.0 setuptools==81.0.0 --force-reinstall
``` 

🧩 Usage

1️⃣ Train a PINN for a simple ODE
```
python train.py
```

Displays convergence in the terminal

TensorBoard logs saved in runs/

Launch TensorBoard:
```
tensorboard --logdir=runs
``` 
Then open your browser at: http://localhost:6006

2️⃣ Visualization

The train.py script plots the PINN solution versus the analytical solution when available.

For ODE systems or integro-differential equations, the visualization is adapted to the problem’s dimensionality.

🧠 Code Structure

- model.py: configurable MLP network for PINNs
- train.py: training loop with loss based on the equation and initial/boundary conditions
- utils.py: helper functions for integration, normalization, quadrature, etc. (optional)