import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import PINN

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

# ---- TensorBoard ----
writer = SummaryWriter("runs/pinn_ode_demo")

# ---- Boucle d'entraînement ----
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Prédiction
    y_pred = model(x)
    
    # Calcul de dy/dx via autograd
    dy_dx = torch.autograd.grad(
        y_pred, x, torch.ones_like(y_pred), create_graph=True
    )[0]
    
    # ODE loss: dy/dx + y = 0
    ode_loss = torch.mean((dy_dx + y_pred)**2)
    
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

writer.close()

# ---- Visualisation ----
x_plot = x.detach().cpu()
y_plot = model(x).detach().cpu()
y_true = torch.exp(-x_plot)  # solution analytique

plt.plot(x_plot, y_plot, label="PINN")
plt.plot(x_plot, y_true, "--", label="Exact")
plt.legend()
plt.show()