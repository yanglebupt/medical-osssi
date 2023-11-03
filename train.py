
import torch
import torch.nn as nn
from models.annet import ANNet
from models.annet_2 import ANNet_2
from models.nnet import NNet

device = 'cpu'
start_lr = 0.0001
loss_fn = nn.CrossEntropyLoss()

def create_optimizer(model):
    optimizer=torch.optim.Adam(model.parameters(),
                            lr=start_lr,
                            betas=(0.9,0.999),
                            eps=1e-8)
    return optimizer

def create_model_by_name(name):
    if name=="nnet":
        model = NNet(ipt=45, end=2, start=128, 
                   block=5, blocksize=3, 
                   dropout=nn.Dropout(0.7)
                  ).to(device)
    elif name=="annet":
        model = ANNet([45,128,256,128,64,32,16,4,2]).to(device)
    elif name=="annet_2":
        model = ANNet_2([45,128,256,128,64,32,16,4,2]).to(device)
    else:
        pass
    return model


def train(device, dataloader, model, loss_fn, optimizer,isprint=False):
    correct=0
    error=0
    total=0
    for batch, (X, l) in enumerate(dataloader):
        l = l.long()
        n = l.shape[0]
        X,l = X.to(device),l.to(device)
        # 前向反馈
        pred = model(X)
        # 计算误差
        loss = loss_fn(pred,l)
        # 开始优化网络权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error+=loss.item()
        
        # 计算准确率
        p=torch.max(nn.functional.softmax(pred,dim=1),1)[1]
        p=p.to(device)
        correct+=(p == l).sum()
        total+=n
        if isprint:
            print(str(n)+"中的正确个数:"+str((p == l).sum().cpu()))
    return error/(batch+1),correct/total

def test(device,dataloader,model,loss_fn,isprint=False):
    correct=0
    error=0
    total=0
    
    for batch,(X,l) in enumerate(dataloader):
        l = l.long()
        n = l.shape[0]
        # 将数据传送至设备里
        X,l = X.to(device),l.to(device)
        # 前向反馈
        outputs = model(X)
        # 计算误差
        loss = loss_fn(outputs, l)
        error+=loss.item()
        # 计算准确率
        p=torch.max(nn.functional.softmax(outputs,dim=1),1)[1]
        p=p.to(device)
        correct+=(p == l).sum()
        total+=n
        if isprint:
            print(str(n)+"中的正确个数:"+str((p == l).sum().cpu()))
    return error/(batch+1),correct/total
