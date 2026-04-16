import torch
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np


class VisualizationMixin:
    @staticmethod
    def fig_to_tensor(fig):
        """Convert matplotlib figure to torch tensor for tensorboard."""
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = torch.tensor(np.array(plt.imread(buf))).permute(2, 0, 1)
        plt.close(fig)
        return img