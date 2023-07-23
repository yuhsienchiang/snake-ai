import os
import torch
import torch.nn as nn


class MLP_QNet(nn.Module):
    def __init__(self, net_arch: list[tuple[int, int]]) -> None:
        super(MLP_QNet, self).__init__()

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else "cpu"
        )

        self.flatten_layer = nn.Flatten().to(self.device)
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim).to(self.device)
                for (in_dim, out_dim) in net_arch
            ]
        )

        self.activation = nn.ReLU().to(self.device)
        self.layer_num = len(net_arch)

    def forward(self, x):
        # flatten the input tensor
        x = self.flatten_layer(x)

        # pass through linear layers
        for linear_layer_idx, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)

            # pass through activation func except the last layers
            if linear_layer_idx < self.layer_num - 1:
                x = self.activation(x)

        return x

    def save_model(self, file_name="mlp_qnet.pth"):
        model_folder_path = "./trained_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
