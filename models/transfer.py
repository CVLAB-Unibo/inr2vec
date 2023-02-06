from torch import Tensor, nn


class Transfer(nn.Module):
    def __init__(self, emb_dim: int, num_layers: int) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(emb_dim, emb_dim))

            if i != num_layers - 1:
                layers.append(nn.BatchNorm1d(emb_dim))
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
