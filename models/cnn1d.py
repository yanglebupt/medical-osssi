import torch
import torch.nn as nn

class CNN1d(nn.Module):
    def __init__(self, dropout=nn.Dropout(0.5), drop_block=nn.Dropout2d(0.1), isHalf=False):
        super(CNN1d, self).__init__()
        self.dropout = dropout
        self.drop_block = drop_block
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(1, 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ])
        
        self.layer2 = nn.Sequential(*[
            nn.Conv2d(4, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            self.drop_block,
            nn.ReLU()
        ])
        
        self.layer3 = nn.Sequential(*[
            nn.Conv2d(16, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            self.drop_block,
            nn.ReLU()
        ]) if not isHalf else None
        
        self.layer4 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            self.drop_block,
            nn.ReLU()
        ]) if not isHalf else None
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(*[
            nn.Linear(64,16),
            nn.LeakyReLU(0.01),
            self.dropout,
            
            nn.Linear(16,4),
            nn.LeakyReLU(0.01),
            self.dropout,
            
            nn.Linear(4,2)
        ]) if not isHalf else nn.Sequential(*[
            nn.Linear(16,4),
            nn.LeakyReLU(0.01),
            self.dropout,
            
            nn.Linear(4,2)
        ])
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        x = self.global_pool(x)
        batch, feas = x.shape[0],x.shape[1]
        x = x.view((batch, feas))
        x = self.fc(x)
        return x