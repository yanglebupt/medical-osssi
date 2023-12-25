import torch
import torch.nn as nn
from train import device
from constants import cls_header_index, headers_2

class GatedLinearUnit(nn.Module):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(units, units)
        self.sigmoid = nn.Sequential(*[
            nn.Linear(units, units),
            nn.Sigmoid()
        ])

    def forward(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
    

class GatedResidualNetwork(nn.Module):
    def __init__(self, feas, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = nn.Sequential(*[
            nn.Linear(feas, units),
            nn.ELU()
        ])
        self.linear_dense = nn.Linear(units, units)
        self.dropout = nn.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = nn.LayerNorm(units)
        self.project = nn.Linear(feas, units)

    def forward(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class VariableSelection(nn.Module):
    """
    num_features: 特征数量 D
    feas: 原始特征维度 F
    units: 变换后的特征维度 U
    """
    def __init__(self, num_features, feas, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.num_features = num_features
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(feas, units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(feas*num_features, units, dropout_rate)
        self.softmax = nn.Sequential(*[
            nn.Linear(units, num_features),
            nn.Softmax(dim=1)
        ])

    def forward(self, inputs):   # [B,D,F]
        B = inputs.shape[0]
        v = inputs.view((B,-1))
        v = self.grn_concat(v)  # [B, D*F] --> [B, U]
        v = self.softmax(v).reshape((B,1,-1))  # [B, U] --> [B, D] --> [B, 1, D]

        x = []
        for idx in range(self.num_features):
            after_grn = self.grns[idx](inputs[:,idx,:])
            x.append(after_grn.reshape((B,1,-1))) # [B, F] ---> [B,1,U]
        x = torch.cat(x, dim=1)  # [B,D,U]

        # [B,1,D]*[B,D,U]  --> [B, 1, U]
        outputs = torch.squeeze(torch.matmul(v, x), dim=1)  # [B, U]
        return outputs
    


class FeatureEmbedding(nn.Module):
    def __init__(self, ipt=1, emb_dim=8, cls_token=False):
        super(FeatureEmbedding, self).__init__()
        self.cls_features_embedding = [nn.Embedding(6, emb_dim) for i in range(len(headers_2))]
        self.embedding = nn.Linear(ipt, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) if cls_token else None
        
    def forward(self, x):
        B,D = x.shape  # [B, D]
        emb_features=[]
        for i in range(D):
            if i in cls_header_index:
                v = torch.LongTensor(x[:,i].cpu().numpy().astype(int)).to(device)
                emb_feature = self.cls_features_embedding[cls_header_index.index(i)](v).reshape((B,1,-1))
            else:
                emb_feature = self.embedding(x[:,i].reshape((-1,1))).reshape((B,1,-1))  # [B,1,F]
            emb_features.append(emb_feature)
        
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            return torch.cat((cls_token, torch.cat(emb_features, dim=1)), dim=1)  # [B,D+1,F]  D 是特征个数，F 特征嵌入维度
        else:
            return  torch.cat(emb_features, dim=1)  # [B,D,F]  D 是特征个数，F 特征嵌入维度
    
class TFEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, norm=nn.LayerNorm, dropout=0.3, act=nn.GELU()):
        super(TFEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation=act)
        self.tf = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.tf(x)
        return x


"""
just_transformer: 只使用 transformer encoder
    - cls_token 用 token 分类
    - 否则直接全连接
use_vs: 使用变量选择器，但不使用 transformer encoder
use_tf_vs  使用变量选择器，使用 transformer encoder
    - tf_vs_shortcut 是否进行残差连接
只使用 变量选择器
"""
class TF(nn.Module):
    def __init__(self, ipt=1, 
                 num_features=45,
                 units=64,
                 emb_dim=64, 
                 blocknums=3, 
                 d_models=[64,64,64], 
                 headers=[4,4,4], 
                 num_layers=[3,3,3], 
                 dropout=0.3, 
                 act=nn.GELU(), 
                 norm=nn.LayerNorm,
                 cls_token=False,
                 just_transformer=True,
                 use_vs=True,
                 use_tf_vs=True,
                 tf_vs_shortcut=True,
                 use_global_pool=True,
                 pool_fea=True
                ):
        super(TF, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.norm = norm
        self.act = act
        self.cls_token = cls_token
        self.just_transformer = just_transformer
        self.use_vs = use_vs
        self.use_tf_vs = use_tf_vs
        self.tf_vs_shortcut = tf_vs_shortcut
        self.use_global_pool = use_global_pool
        self.pool_fea = pool_fea
        self.featureEmbedding = FeatureEmbedding(ipt, emb_dim, cls_token)
        tf_blocks = []
        for i in range(blocknums):
            tf_blocks.append(TFEncoder(d_model=d_models[i], nhead=headers[i], num_layers=num_layers[i], norm=norm, dropout=dropout, act=act))
        self.tf_blocks = nn.Sequential(*tf_blocks)
        
        self.variableSelection = VariableSelection(num_features, emb_dim, units, dropout_rate=dropout)  # 45-->units
        self.variableSelection1 = VariableSelection(units, 1, 2*units, dropout_rate=dropout+0.2) 
        self.variableSelection2 = VariableSelection(2*units, 1, 4*units, dropout_rate=dropout+0.2*2)
        self.variableSelection3 = VariableSelection(4*units, 1, 2*units, dropout_rate=dropout+0.2)
        self.variableSelection4 = VariableSelection(2*units, 1, units, dropout_rate=dropout)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.vs_head = nn.Linear(units, 2) 
        self.fea_head = nn.Linear(num_features, 2) 
        self.cls_head = nn.Linear(emb_dim, 2) 
        self.tf_head = nn.Sequential(*[
            nn.Linear(emb_dim*num_features,128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            
            nn.Linear(128,16),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            
            nn.Linear(16,4),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
        
            nn.Linear(4,2)
        ])

    def forward(self, x):
        B = x.shape[0]
        emb_feas = self.featureEmbedding(x)
        if self.just_transformer or self.use_tf_vs:
            x = self.tf_blocks(emb_feas)   # [B,D+1,F]
        if self.just_transformer:
            if self.cls_token:
                return self.cls_head(x[:,-1,:])
            else:
                if self.use_global_pool:
                    if self.pool_fea:
                        x = self.global_pool(x.view(B, x.shape[2], x.shape[1]))   # [B, F, 1]
                        return self.cls_head(x.view(B,-1))
                    else:
                        x = self.global_pool(x)   # [B, D, 1]
                        return self.fea_head(x.view(B,-1))
                else:
                    return self.tf_head(x.view(B,-1))
        if self.use_vs:
            x = self.variableSelection(emb_feas)
            x = x.view((B,-1,1))
            x = self.variableSelection1(x)
            x = x.view((B,-1,1))
            x = self.variableSelection2(x)
            x = x.view((B,-1,1))
            x = self.variableSelection3(x)
            x = x.view((B,-1,1))
            x = self.variableSelection4(x)
                
        if self.use_tf_vs:
            if self.tf_vs_shortcut:
                x = x + emb_feas
            x = self.variableSelection(x)
        return self.vs_head(x)
    def __init__(self, ipt=1, 
                 num_features=45,
                 units=64,
                 emb_dim=64, 
                 blocknums=3, 
                 d_models=[64,64,64], 
                 headers=[4,4,4], 
                 num_layers=[3,3,3], 
                 dropout=0.3, 
                 act=nn.GELU(), 
                 norm=nn.LayerNorm,
                 cls_token=False
                ):
        super(TF, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.norm = norm
        self.act = act
        self.cls_token = cls_token
        self.featureEmbedding = FeatureEmbedding(ipt, emb_dim, cls_token)
        tf_blocks = []
        for i in range(blocknums):
            tf_blocks.append(TFEncoder(d_model=d_models[i], nhead=headers[i], num_layers=num_layers[i], norm=norm, dropout=dropout, act=act))
        self.tf_blocks = nn.Sequential(*tf_blocks)
        
        self.variableSelection = VariableSelection(num_features, emb_dim, units, dropout_rate=dropout)
        
        self.head = nn.Linear(units, 2) 

    def forward(self, x):
        B = x.shape[0]
        x = self.featureEmbedding(x)
        x = self.tf_blocks(x)   # [B,D+1,F]
        x = self.variableSelection(x)
        return self.head(x)
    
"""
just_transformer: 只使用 transformer encoder
    - cls_token 用 token 分类
    - 否则直接全连接
use_vs: 使用变量选择器，但不使用 transformer encoder
use_tf_vs  使用变量选择器，使用 transformer encoder
    - tf_vs_shortcut 是否进行残差连接
"""

tf_st_vs = {
    "just_transformer":False,
    "cls_token":False,
    "use_vs":False,
    "use_tf_vs":True,
    "tf_vs_shortcut":True
}
tf_vs = {
    "just_transformer":False,
    "cls_token":False,
    "use_vs":False,
    "use_tf_vs":True,  # 使用单层 vs
    "tf_vs_shortcut":False
}
l_vs = {
    "just_transformer":False,
    "cls_token":False,
    "use_vs":True,   # 使用多层 vs，无残差连接
    "use_tf_vs":False,
    "tf_vs_shortcut":False
}
tf_cls = {
    "just_transformer":True,
    "cls_token":True,
    "use_vs":False,
    "use_tf_vs":False,
    "tf_vs_shortcut":False
}
tf_fc = {
    "just_transformer":True,
    "cls_token":False,
    "use_vs":False,
    "use_tf_vs":False,
    "tf_vs_shortcut":False,
    "use_global_pool":True,
    "pool_fea":False
}

all_tf_dict = {
    "tf_st_vs":tf_st_vs,
    "tf_vs":tf_vs,
    "l_vs":l_vs,
    "tf_cls":tf_cls,
    "tf_fc":tf_fc
}