import torch
import torch.nn as nn
class ANNet_2(nn.Module):
    def __init__(self, layer_n, dropout=nn.Dropout(0.5), norm=nn.BatchNorm1d):
        super(ANNet_2, self).__init__()
        self.dropout = dropout
        self.norm = norm
        self.layers = nn.ModuleList([dropout])
        self._gen_layers(layer_n)

    def _gen_layers(self, layer_n):
        for i in range(len(layer_n)-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layer_n[i], layer_n[i+1]), self.norm(layer_n[i+1]), nn.LeakyReLU(0.01)
                )
            )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x