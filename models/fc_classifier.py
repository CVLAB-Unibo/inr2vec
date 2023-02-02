from typing import List

from torch import Tensor, nn


class FcClassifier(nn.Module):
    def __init__(self, layers_dim: List[int], num_classes: int) -> None:
        super().__init__()

        layers = []
        if len(layers_dim) > 1:
            for i in range(len(layers_dim) - 1):
                layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
                layers.append(nn.BatchNorm1d(layers_dim[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout())
        layers.append(nn.Linear(layers_dim[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
