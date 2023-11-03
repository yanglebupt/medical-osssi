import torch.nn as nn
import torch

def _gen_linear(pre_nums, next_nums, dropout=None, active=None):
    res = [nn.Linear(pre_nums, next_nums), dropout] if dropout else [nn.Linear(pre_nums, next_nums)]
    if active:
        res.append(active)
    
    return nn.Sequential(*res)

class ResNNet(nn.Module):
    def __init__(self, ipt, blocksize=3):
        super(ResNNet, self).__init__()
        self.layers = nn.ModuleList([_gen_linear(ipt, ipt) for i in range(blocksize)])
        
    def forward(self, x):
        temp = x
        for l in self.layers:
            temp = l(temp)
        return temp + x

class NNet(nn.Module):
    def __init__(self, ipt, end, start=128, block=5, blocksize=3, dropout=nn.Dropout(0.7), active=nn.LeakyReLU(0.01), norm=nn.BatchNorm1d):
        super(NNet, self).__init__()
        self.layers = nn.ModuleList([dropout, _gen_linear(ipt,start, None, None), norm(start), active])
        for i in range(block):
            self.layers.append(ResNNet(ipt = start, blocksize = blocksize))
            self.layers.append(
                nn.Sequential(norm(start), active)
            )
        for seq in [
            _gen_linear(start, 64, norm(64), active),
            _gen_linear(64, 16, norm(16), active),
            _gen_linear(16, end)
        ]:
            self.layers.append(seq)
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x