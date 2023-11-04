import torch
import torch.nn as nn
import numpy as np

def test_proba(device,dataloader,model,loss_fn, conv1d=False):
    error=0
    fin_probas=None
    fin_ls=None
    for batch,(X,l) in enumerate(dataloader):
        l = l.long()
        n = l.shape[0]
        if conv1d:
            X = X.reshape((n,1,-1,1))
        # 将数据传送至设备里
        X,l = X.to(device),l.to(device)
        # 前向反馈
        outputs = model(X)
        # 计算误差
        loss = loss_fn(outputs, l)
        error+=loss.item()
        
        probas = nn.functional.softmax(outputs,dim=1).detach().cpu()
        l = l.detach().cpu()
        
        fin_probas = probas if fin_probas is None else np.concatenate([fin_probas, probas], axis=0)
        fin_ls = l if fin_ls is None else np.concatenate([fin_ls, l], axis=0)
        
    return error/(batch+1), fin_probas, fin_ls