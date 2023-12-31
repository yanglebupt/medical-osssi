import torch
import torch.nn as nn

class ANNet(nn.Module):
    def __init__(self, layer_n, dropout=nn.Dropout(0.5)):
        super(ANNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self._gen_layers(layer_n)

    def _gen_layers(self, layer_n):
        for i in range(len(layer_n)-1):
            self.layers.append(
                nn.Sequential(
                    self.dropout, nn.Linear(layer_n[i], layer_n[i+1]), nn.LeakyReLU(0.01)
                )
            )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x