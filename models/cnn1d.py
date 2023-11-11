import torch.nn as nn

class CNN1d(nn.Module):
    def __init__(self, dropout=nn.Dropout(0.5), drop_block=nn.Dropout2d(0.1), dropoutFirst=nn.Dropout2d(0.5),isHalf=False):
        super(CNN1d, self).__init__()
        self.dropout = dropout
        self.drop_block = drop_block
        self.dropoutFirst = dropoutFirst
        
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(1, 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ])
        
        list2 = [
            nn.Conv2d(4, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
        ]
        if self.drop_block is not None:
            list2.append(self.drop_block)
        list2.append(nn.ReLU())
        self.layer2 = nn.Sequential(*list2)
        
        list3 = [
            nn.Conv2d(16, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        ]
        if self.drop_block is not None:
            list3.append(self.drop_block)
        list3.append(nn.ReLU())
        self.layer3 = nn.Sequential(*list3) if not isHalf else None
        
        
        list4 = [
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        ]
        if self.drop_block is not None:
            list4.append(self.drop_block)
        list4.append(nn.ReLU())
        self.layer4 = nn.Sequential(*list4) if not isHalf else None
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        if not isHalf:
            fcList = [
                nn.Linear(64,16),
                nn.LeakyReLU(0.01),
            ]
            if self.dropout is not None:
                fcList.append(self.dropout)
            fcList+=[
                nn.Linear(16,4),
                nn.LeakyReLU(0.01),
            ]
            if self.dropout is not None:
                fcList.append(self.dropout)
            fcList.append(nn.Linear(4,2))
        else:
            fcList = [
                nn.Linear(16,4),
                nn.LeakyReLU(0.01),
            ]
            if self.dropout is not None:
                fcList.append(self.dropout)
            fcList.append(nn.Linear(4,2))
        
        self.fc = nn.Sequential(*fcList)
        
        
    def forward(self, x):
        if self.dropoutFirst is not None:
            x = self.dropoutFirst(x)
            
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