import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from model import PINN
from ode import ode_lotka_voltera, Params, validation_lotka_voltera

# ---- Hyperparamètres ----
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
epochs = 2000
N = 100  # nombre de points

# ---- Génération des points ----
x = torch.linspace(0, 2, N).view(-1,1).to(device)  # intervalle [0,2]
x.requires_grad = True
y0 = 1.0  # condition initiale

# ---- Modèle ----
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----- ODE Parameters----
params_ode = Params(alpha=1., beta=2.)
# ---- TensorBoard ----
writer = SummaryWriter("runs/lotka_eval")

# ---- Boucle d'entraînement ----
global_step = 0
for epoch in range(epochs):
    global_step += 1
    optimizer.zero_grad()
    
    # Prédiction
    y_pred = model(x)
    
    # Calcul de dy/dx via autograd
    dy_dx = torch.autograd.grad(
        y_pred, x, torch.ones_like(y_pred), create_graph=True
    )[0]
    
    dx_dt = ode_lotka_voltera(y_pred, params= params_ode)
    
    # ODE loss: dy/dx + y = 0
    ode_loss = torch.mean((dy_dx - dx_dt)**2)
    
    # Condition initiale
    ic_loss = (model(torch.tensor([[0.0]], device=device)) - y0)**2
    
    # Loss totale
    loss = ode_loss + ic_loss
    
    loss.backward()
    optimizer.step()
    
    # TensorBoard
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        writer.add_scalar("Loss/train", loss.item(), epoch)

# Mode evaluation
model.eval()
sol = validation_lotka_voltera(t_span = (0, 20),z0= [10, 5], params=params_ode)
x_true = sol.y[0]  # proies
y_true = sol.y[1]  # prédateurs
t_true = sol.t
x_true_tensor = torch.tensor(x_true, dtype=torch.float32).unsqueeze(1)  # [200,1]
y_true_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)  # [200,1]
t_true_tensor = torch.tensor(t_true, dtype=torch.float32).unsqueeze(1)  # [200,1]
with torch.no_grad():
    # Exemple : prédire sur t_true_tensor
    y_pred_eval = model(t_true_tensor)  # [200,2]

    # Calcul d'une erreur MSE par rapport aux données simulées
    mse_eval = torch.mean((y_pred_eval[:,0] - x_true_tensor.squeeze())**2 +
                          (y_pred_eval[:,1] - y_true_tensor.squeeze())**2)

    # Écriture dans TensorBoard
    writer.add_scalar("Eval/MSE", mse_eval.item(), global_step)
fig, ax = plt.subplots()
ax.plot(t_true, x_true, label="Proies (SciPy)")
ax.plot(t_true, y_true, label="Prédateurs (SciPy)")
ax.plot(t_true, y_pred_eval[:,0].numpy(), '--', label="Proies (NN)")
ax.plot(t_true, y_pred_eval[:,1].numpy(), '--', label="Prédateurs (NN)")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Population")

# Convertir en image pour TensorBoard
buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
img = torch.tensor(np.array(plt.imread(buf))).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
writer.add_image("Trajectories", img[0], global_step)
plt.close(fig)

writer.close()

# ---- Visualisation ----
x_plot = x.detach().cpu()
y_plot = model(x).detach().cpu()
y_true = torch.exp(-x_plot)  # solution analytique

plt.plot(x_plot, y_plot, label="PINN")
plt.plot(x_plot, y_true, "--", label="Exact")
plt.legend()
plt.show()