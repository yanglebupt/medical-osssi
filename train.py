
import torch
import torch.nn as nn
from models.annet import ANNet
from models.annet_2 import ANNet_2
from models.nnet import NNet
from models.cnn1d import CNN1d
from models.tf import TF, all_tf_dict
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = 'cpu'
start_lr = 0.0001


def create_loss_fn(weight_name="balanced", train_labels=None, device="cpu"):
  if weight_name is None:
      return nn.CrossEntropyLoss()  
  else:
    weights = torch.FloatTensor(calc_classes_weights(train_labels, method=weight_name))
    return nn.CrossEntropyLoss(weight=weights.to(device))    
  
def create_loss_fn_weights(weights, device="cpu"):
  if weights is None:
      return nn.CrossEntropyLoss()  
  else:
    return nn.CrossEntropyLoss(weight=weights.to(device))    

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
                  )
    elif name=="ann":
        model = ANNet([45,128,256,128,64,32,16,4,2], norm=None)
    elif name=="ann-norm":
        model = ANNet_2([45,128,256,128,64,32,16,4,2])
    elif name.startswith("cnn1d-half"):
        model = CNN1d(dropout=None, drop_block=None, dropoutFirst=None, isHalf=True)
    elif name=="cnn1d-fit":
        model = CNN1d(dropout=nn.Dropout(0.7), drop_block=nn.Dropout2d(0.3), dropoutFirst=None)
    elif name.startswith("tf"):
        model = TF(
            num_features=46,
            units=16,
            emb_dim=16, blocknums=1, d_models=[16], headers=[2], num_layers=[1], dropout=0.5,
            **all_tf_dict["tf-fc"]
        )
    else:
        pass
    return model


def calc_classes_weights(labels, method="balanced"):
    classes = np.unique(labels)
    nums_list=[len(np.where(labels==cl)[0]) for cl in classes]
    print(nums_list)
    if method=="balanced":
        return compute_class_weight("balanced", classes=classes, y=labels)
    elif method=="max":
        # 即用类别中最大样本数量除以当前类别样本的数量，作为权重系数
        max_nums = np.max(nums_list)
        return [max_nums/nums for nums in nums_list]
    elif method=="reciprocal":
        return [1/nums for nums in nums_list]
    else:
        pass

def train(device, dataloader, model, loss_fn, optimizer=None, train=True, conv1d=False, isprint=False, useproba=False, weights=None):
    correct=0
    error=0
    total=0
    
    fin_probas=None
    fin_ls=None
    
    for batch, (X, l) in enumerate(dataloader):
        l = l.long()
        n = l.shape[0]
        if conv1d:
            X = X.reshape((n,1,-1,1))
        X,l = X.to(device),l.to(device)
        # 前向反馈
        outputs = model(X)

        # 计算误差
        loss = loss_fn(outputs,l)
        
        if train:
            # 开始优化网络权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        error+=loss.item()
        
        # 计算准确率
        if weights is None:
            probas = nn.functional.softmax(outputs,dim=1)
        else:
            probas = nn.functional.softmax(weights*outputs,dim=1)
            
        p=torch.max(probas,1)[1]
        p=p.to(device)
        correct+=(p == l).sum()
        total+=n
        if isprint:
            print(str(n)+"中的正确个数:"+str((p == l).sum().cpu()))
            
        if useproba:
            probas = probas.detach().cpu()
            l = l.detach().cpu()
            fin_probas = probas if fin_probas is None else np.concatenate([fin_probas, probas], axis=0)
            fin_ls = l if fin_ls is None else np.concatenate([fin_ls, l], axis=0)
    
    return error/(batch+1),correct/total, fin_probas, fin_ls