import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_QNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MLP_QNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)
        return self.output_layer(x)

    def save_model(self, file_name="mlp_qnet.pth"):
        file_name = os.path.join("./", file_name)
        torch.save(self.state_dict(), file_name)
