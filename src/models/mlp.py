import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_QNet(nn.Module):
    def __init__(self, input_dim: int, hidden_1_dim: int, hidden_2_dim: int, output_dim: int) -> None:
        super(MLP_QNet, self).__init__()

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else "cpu"
        )

        self.input_layer = nn.Flatten()
        self.hidden_1_layer = nn.Linear(input_dim, hidden_1_dim)
        self.hidden_2_layer = nn.Linear(hidden_1_dim, hidden_2_dim)
        self.output_layer = nn.Linear(hidden_2_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_1_layer(x)
        x = F.relu(x)
        x = self.hidden_2_layer(x)
        x = F.relu(x)
        return self.output_layer(x)

    def save_model(self, file_name="mlp_qnet.pth"):
        file_name = os.path.join("./", file_name)
        torch.save(self.state_dict(), file_name)
